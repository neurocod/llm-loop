// ---------------------------------------------------------------------------
// The core broker: heavy runs (a test suite, a compiler, a probe batch) LEASE
// CPU cores instead of each taking all of them blind.
//
// Pure bookkeeping, no sockets and no clock of its own: Registry owns the wire
// and feeds the measured load (`addIdleSample`), so `AgentCounter --selftest`
// can drive every rule here with made-up machines.
//
// The rules, each argued where it is applied:
//   * free cores = floor(total - leased - foreign), a lease reserved whole
//     and foreign load measured apart from it — `freeCores`, `addSample`;
//   * first come, first served, a queue that is never overtaken, served from
//     its head only — `settle`;
//   * a running lease is never shrunk — `settle`;
//   * the head, when it cannot get its `min`, waits up to its own `wait` and
//     then is granted `min` anyway, marked forced — `settle`;
//   * an optional budget caps what all leases together may hold, without
//     taking back what is already held — `setBudget`.
// ---------------------------------------------------------------------------
#pragma once

#include <QDateTime>
#include <QString>
#include <QVector>

#include <algorithm>

// One running lease. `cores` never changes after the grant.
struct CoreLease {
	quint64 id = 0;
	QString label;
	int cores = 0;
	QDateTime granted;
	bool forced = false;  // granted past its wait, over the free cores
};

// One request waiting in the queue.
struct CoreRequest {
	quint64 id = 0;
	QString label;
	int want = 1;
	int min = 1;
	QDateTime asked;
	qint64 waitMs = 0;  // after this the request is granted `min`, forced
};

// What `settle` decided for one request: Registry sends it to its client.
struct CoreGrant {
	quint64 id = 0;
	int cores = 0;
	bool forced = false;
	qint64 waitedMs = 0;
	int freeBefore = 0;  // the free figure the decision was made on
};

class CoreBroker {
public:
	// Samples averaged into the measured idle fraction, one per second
	// (Registry's meter tick), so a 5 s sliding window. Why 5: a single
	// second is one link step or one GC pause of a neighbour and would grant
	// or refuse on noise; 30 s would still count a load that ended half a
	// minute ago and miss one that just began. The leased runs this exists
	// for last minutes, so reacting within 5 s costs them nothing.
	static constexpr int kWindowSamples = 5;

	explicit CoreBroker(int machineCores);

	// The cores leases may share: the budget when one is set, else the
	// machine. Every grant, clamp and badge figure reads this one.
	int total() const { return m_budget > 0 ? std::min(m_budget, m_machine) : m_machine; }
	// The machine's logical cores: what the meter's idle fraction is OF.
	int machine() const { return m_machine; }
	// The operator's ceiling on all leases together, 0 = none (`/testSlow`:
	// the human is using the machine, heavy runs keep to 2 cores). It is a cap
	// on LEASES, not a smaller machine: foreign load is still measured against
	// the whole machine, because the human's own load is exactly what the
	// budget makes room for — subtracting it from 2 as well would leave
	// nothing and force every run. A budget below what is already leased
	// takes nothing back (a lease is baked into a child's `-j`, `settle`); it
	// only grants nothing until the leases fall under it.
	void setBudget(int cores) { m_budget = std::max(0, cores); }
	int budget() const { return m_budget; }

	// One meter interval: the idle fraction of the whole machine, in [0, 1]
	// (clamped: a meter rounding past either end must not mint or destroy
	// cores), and how many cores of the busy rest were the LEASES' own use,
	// each lease counted up to its size (Registry::meterStep measures it).
	//
	// What is stored is the interval's FOREIGN load, busy - own, not the idle
	// fraction, and that is what makes a handoff whole: the sample was taken
	// while the departing lease ran, so its cores are in `own`, not in the
	// foreign figure, and the moment it is released they are free — the
	// window still reading "busy" no longer matters. Storing idle instead
	// gave a waiter behind a 16-core run ~3 cores at the next tick, for good.
	void addSample(double idleFraction, double ownCores);
	// `addSample` for a meter that cannot see the leases' use: each running
	// lease is assumed to burn all it holds. That makes the foreign figure a
	// LOWER bound (a leaseholder in a single-threaded phase hides foreign load
	// in its unused cores), which is what a lease without a job (Registry,
	// `job` on the wire) gets.
	void addIdleSample(double idleFraction) { addSample(idleFraction, leased()); }
	// No sample yet = no grant yet: right after the counter starts there is
	// nothing to subtract foreign load FROM, and granting all cores to the
	// first comer on that ignorance is the very blindness this replaces.
	bool measured() const { return !m_idle.isEmpty(); }
	// Mean of the window, in cores. 0 when unmeasured. For the status reader.
	double idleCores() const;
	// Mean of the window's foreign load, in cores: what was busy beyond the
	// leases' own use (Chrome, an IDE, the human, a run nobody leased, and a
	// lease's use BEYOND its size). 0 when unmeasured.
	double foreignCores() const;
	int leased() const;
	// floor(min(machine - foreign, total) - leased), never below 0: without a
	// budget, floor(machine - leased - foreign). A lease is reserved
	// WHOLE, whatever it burns right now: a leaseholder between phases will
	// burn what it holds again, and a just-granted child has not started at
	// all. So an 8-core lease using 1 core beside 4 cores of foreign load
	// leaves 16 - 8 - 4 = 4 free, not the 11 the meter sees idle. No reserve
	// is kept back for the human in v1: nothing has measured one.
	int freeCores() const;

	// Queue a request (validated by the caller: 1 <= min <= want). Its
	// counts are clamped to `total` when it is decided, not here: a budget set
	// or lifted while it waits applies to it. Grants happen only in `settle`.
	void request(const CoreRequest& request);
	// The client left: its lease or its place in the queue goes.
	void release(quint64 id);
	// Grant what can be granted now, in order (see the .cpp for the rules).
	QVector<CoreGrant> settle(const QDateTime& now);

	const QVector<CoreLease>& leases() const { return m_leases; }
	const QVector<CoreRequest>& queue() const { return m_queue; }

private:
	int m_machine;
	int m_budget = 0;
	QVector<double> m_idle;    // idle fractions, oldest first
	QVector<double> m_foreign;  // foreign cores of the same intervals
	QVector<CoreLease> m_leases;
	QVector<CoreRequest> m_queue;
};

// The machine's logical processor count, across processor groups.
int logicalCoreCount();

// Whole-machine CPU meter (Windows GetSystemTimes). Each `sample` returns the
// idle fraction since the previous call, or a negative value when there is no
// previous call (the first) or no meter (not Windows); `intervalSeconds` is
// the wall time that fraction covers (CPU time over all cores / cores).
class CpuMeter {
public:
	double sample();
	double intervalSeconds() const { return m_intervalS; }

private:
	bool m_primed = false;
	quint64 m_idle = 0;
	quint64 m_total = 0;
	double m_intervalS = 0.0;
};

// A lease's job object (Windows), opened by name for QUERY only: the CPU
// time of everything the leased child and its descendants burned. The
// handle also keeps the job alive, so a client's kill-on-close job ends its
// child only once the broker has let go of it as well (Registry closes it as
// it releases the lease). Null when the name opens nothing.
void* openJob(const QString& name);
void closeJob(void* job);
// Total user + kernel CPU seconds of the job, or a negative value.
double jobCpuSeconds(void* job);
