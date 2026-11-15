// ---------------------------------------------------------------------------
// The core broker: heavy runs (a test suite, a compiler, a probe batch) LEASE
// CPU cores instead of each taking all of them blind.
//
// Pure bookkeeping, no sockets and no clock of its own: Registry owns the wire
// and feeds the measured load (`addIdleSample`), so `AgentCounter --selftest`
// can drive every rule here with made-up machines.
//
// The rules, each argued where it is applied:
//   * free cores = floor(min(total - leased, total * idle))  — `freeCores`;
//   * first come, first served, a queue that is never overtaken — `settle`;
//   * a running lease is never shrunk — `settle`;
//   * a request that cannot get its `min` waits, up to its own `wait`, and
//     then is granted `min` anyway, marked forced — `settle`.
// ---------------------------------------------------------------------------
#pragma once

#include <QDateTime>
#include <QString>
#include <QVector>

// One running lease. `cores` never changes after the grant.
struct CoreLease {
	quint64 id = 0;
	QString label;
	int cores = 0;
	QDateTime granted;
	bool forced = false;  // granted past its wait, over the budget
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

	explicit CoreBroker(int totalCores);

	int total() const { return m_total; }

	// The idle fraction of the whole machine over the last sample interval,
	// in [0, 1] (clamped: a meter rounding past either end must not mint or
	// destroy cores).
	void addIdleSample(double idleFraction);
	// No sample yet = no grant yet: right after the counter starts there is
	// nothing to subtract foreign load FROM, and granting all cores to the
	// first comer on that ignorance is the very blindness this replaces.
	bool measured() const { return !m_samples.isEmpty(); }
	// Mean of the window, in cores. 0 when unmeasured.
	double idleCores() const;
	// What is busy beyond the leases: max(0, busy - leased). A lease is a
	// reservation, not a measurement — a leaseholder in a single-threaded
	// phase uses less than it holds — so this is a LOWER bound of the
	// unregistered load (Chrome, an IDE, the human), shown for the status
	// reader; the grant rule does not need it (see `freeCores`).
	double foreignCores() const;
	int leased() const;
	// floor(min(total - leased, idle cores)). Both bounds are needed:
	//   * total - leased alone ignores the unregistered load entirely;
	//   * idle alone double-grants a lease that has not started burning its
	//     cores yet (the child is still starting when the next request comes).
	// It is the same as total - leased - foreign with foreign as above, which
	// is the item's formula; written as a min it cannot go negative. No
	// reserve is kept back for the human in v1: nothing has measured one.
	int freeCores() const;

	// Queue a request (validated by the caller: 1 <= min <= want, both
	// already clamped to total). Grants happen only in `settle`.
	void request(const CoreRequest& request);
	// The client left: its lease or its place in the queue goes.
	void release(quint64 id);
	// Grant what can be granted now, in order (see the .cpp for the rules).
	QVector<CoreGrant> settle(const QDateTime& now);

	const QVector<CoreLease>& leases() const { return m_leases; }
	const QVector<CoreRequest>& queue() const { return m_queue; }

private:
	int m_total;
	QVector<double> m_samples;  // idle fractions, oldest first
	QVector<CoreLease> m_leases;
	QVector<CoreRequest> m_queue;
};

// The machine's logical processor count, across processor groups.
int logicalCoreCount();

// Whole-machine CPU meter (Windows GetSystemTimes). Each `sample` returns the
// idle fraction since the previous call, or a negative value when there is no
// previous call (the first) or no meter (not Windows).
class CpuMeter {
public:
	double sample();

private:
	bool m_primed = false;
	quint64 m_idle = 0;
	quint64 m_total = 0;
};
