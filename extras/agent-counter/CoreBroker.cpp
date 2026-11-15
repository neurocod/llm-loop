#include "CoreBroker.h"

#include <QThread>

#include <algorithm>
#include <cmath>

#ifdef Q_OS_WIN
#include <windows.h>
#endif

CoreBroker::CoreBroker(int totalCores) : m_total(std::max(1, totalCores)) {}

namespace {

double mean(const QVector<double>& values) {
	if (values.isEmpty())
		return 0.0;
	double sum = 0.0;
	for (const double value : values)
		sum += value;
	return sum / values.size();
}

}  // namespace

void CoreBroker::addSample(double idleFraction, double ownCores) {
	const double idle = std::clamp(idleFraction, 0.0, 1.0);
	const double busy = m_total * (1.0 - idle);
	m_idle.append(idle);
	m_foreign.append(std::clamp(busy - ownCores, 0.0, static_cast<double>(m_total)));
	while (m_idle.size() > kWindowSamples) {
		m_idle.removeFirst();
		m_foreign.removeFirst();
	}
}

double CoreBroker::idleCores() const {
	return m_total * mean(m_idle);
}

double CoreBroker::foreignCores() const {
	return mean(m_foreign);
}

int CoreBroker::leased() const {
	int sum = 0;
	for (const CoreLease& lease : m_leases)
		sum += lease.cores;
	return sum;
}

int CoreBroker::freeCores() const {
	if (!measured())
		return 0;
	// A hair of float noise (a mean of five 0.2-core samples = 0.2000...1)
	// must not cost a whole core, hence the epsilon before the floor.
	const double free = std::floor(m_total - leased() - foreignCores() + 1e-6);
	return std::max(0, static_cast<int>(free));
}

void CoreBroker::request(const CoreRequest& request) {
	m_queue.append(request);
}

void CoreBroker::release(quint64 id) {
	m_leases.erase(std::remove_if(m_leases.begin(), m_leases.end(),
	                              [id](const CoreLease& lease) { return lease.id == id; }),
	               m_leases.end());
	m_queue.erase(std::remove_if(m_queue.begin(), m_queue.end(),
	                             [id](const CoreRequest& request) { return request.id == id; }),
	              m_queue.end());
}

QVector<CoreGrant> CoreBroker::settle(const QDateTime& now) {
	QVector<CoreGrant> grants;
	auto grant = [&](const CoreRequest& request, int cores, bool forced, int freeBefore) {
		m_leases.append({request.id, request.label, cores, now, forced});
		grants.append({request.id, cores, forced, request.asked.msecsTo(now), freeBefore});
	};

	// Strictly first come, first served, and ONLY the head is ever decided;
	// the next request is looked at once the head has left the queue.
	//
	// The head gets min(want, free) — greedy, not a proportional share: a
	// share would need SHRINKING running leases when a newcomer arrives, and a
	// lease is a number already baked into a child's command line (`-j 8`),
	// which no message can take back. Nothing is granted on free cores while
	// unmeasured (`measured`).
	//
	// A head that does not fit waits until its own patience runs out and
	// then runs with its `min`, over the free cores, and says so (forced; the
	// badge turns red). The alternative — waiting on — wedges a gate behind a
	// neighbour's long run, and failing it turns load into a red that means
	// nothing; both are worse than one honestly over-committed run. Forced
	// even while unmeasured: a meter that never produced a sample must not
	// hold anyone forever either. A request that fits is never forced, however
	// little patience it asked for: the fit is tried first.
	//
	// Nobody behind the head is decided — not even one whose own patience
	// ran out (it was forced past a waiting head before, and the head's
	// order meant nothing), and not one that would fit (no backfill). The
	// cost is named: cores stay idle while a head too large for them waits,
	// for at most that head's `wait`. Backfill would use them, but it is safe
	// only when a backfilled run can be shown to end before the head could
	// start, and a lease carries no duration: a small run let past the head
	// that outlives the blocker turns the head's clean grant into a forced
	// over-commit — and as long as smalls keep coming (gates are frequent,
	// large mins rare) the large head would be forced EVERY time.
	while (!m_queue.isEmpty()) {
		const CoreRequest head = m_queue.first();
		const int free = freeCores();
		if (measured() && free >= head.min) {
			m_queue.removeFirst();
			grant(head, std::min(head.want, free), false, free);
		} else if (head.asked.msecsTo(now) >= head.waitMs) {
			m_queue.removeFirst();
			grant(head, head.min, true, free);
		} else {
			break;
		}
	}
	return grants;
}

int logicalCoreCount() {
#ifdef Q_OS_WIN
	const DWORD count = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
	if (count > 0)
		return static_cast<int>(count);
#endif
	return std::max(1, QThread::idealThreadCount());
}

double CpuMeter::sample() {
#ifdef Q_OS_WIN
	FILETIME idleTime, kernelTime, userTime;
	if (!GetSystemTimes(&idleTime, &kernelTime, &userTime))
		return -1.0;
	auto ticks = [](const FILETIME& time) {
		return (static_cast<quint64>(time.dwHighDateTime) << 32) | time.dwLowDateTime;
	};
	// Kernel time INCLUDES idle time (GetSystemTimes' documented quirk), so
	// the total is kernel + user, not idle + kernel + user.
	const quint64 idle = ticks(idleTime);
	const quint64 total = ticks(kernelTime) + ticks(userTime);
	const bool primed = m_primed;
	const quint64 idleDelta = idle - m_idle;
	const quint64 totalDelta = total - m_total;
	m_primed = true;
	m_idle = idle;
	m_total = total;
	if (!primed || totalDelta == 0)
		return -1.0;
	// 100 ns ticks summed over every core: per core, that is wall time.
	m_intervalS = static_cast<double>(totalDelta) / 1e7 / logicalCoreCount();
	return static_cast<double>(idleDelta) / static_cast<double>(totalDelta);
#else
	return -1.0;
#endif
}

void* openJob(const QString& name) {
#ifdef Q_OS_WIN
	if (name.isEmpty())
		return nullptr;
	return OpenJobObjectW(JOB_OBJECT_QUERY, FALSE, reinterpret_cast<const wchar_t*>(name.utf16()));
#else
	Q_UNUSED(name);
	return nullptr;
#endif
}

void closeJob(void* job) {
#ifdef Q_OS_WIN
	if (job != nullptr)
		CloseHandle(static_cast<HANDLE>(job));
#else
	Q_UNUSED(job);
#endif
}

double jobCpuSeconds(void* job) {
#ifdef Q_OS_WIN
	JOBOBJECT_BASIC_ACCOUNTING_INFORMATION info{};
	if (job == nullptr
	    || !QueryInformationJobObject(static_cast<HANDLE>(job), JobObjectBasicAccountingInformation,
	                                  &info, sizeof info, nullptr))
		return -1.0;
	// Processes that already exited stay in these totals: a test runner's
	// short-lived workers are counted, not lost between two samples.
	return static_cast<double>(info.TotalUserTime.QuadPart + info.TotalKernelTime.QuadPart) / 1e7;
#else
	Q_UNUSED(job);
	return -1.0;
#endif
}
