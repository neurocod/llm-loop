#include "CoreBroker.h"

#include <QThread>

#include <algorithm>
#include <cmath>

#ifdef Q_OS_WIN
#include <windows.h>
#endif

CoreBroker::CoreBroker(int totalCores) : m_total(std::max(1, totalCores)) {}

void CoreBroker::addIdleSample(double idleFraction) {
	m_samples.append(std::clamp(idleFraction, 0.0, 1.0));
	while (m_samples.size() > kWindowSamples)
		m_samples.removeFirst();
}

double CoreBroker::idleCores() const {
	if (m_samples.isEmpty())
		return 0.0;
	double sum = 0.0;
	for (const double idle : m_samples)
		sum += idle;
	return m_total * sum / m_samples.size();
}

double CoreBroker::foreignCores() const {
	if (!measured())
		return 0.0;
	const double busy = m_total - idleCores();
	return std::max(0.0, busy - leased());
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
	// A hair of float noise (a window of 1.0 samples times 16 = 15.999...)
	// must not cost a whole core, hence the epsilon before the floor.
	const double idle = std::floor(idleCores() + 1e-6);
	return std::max(0, static_cast<int>(std::min<double>(m_total - leased(), idle)));
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

	// Strictly first come, first served, from the head only. The head gets
	// min(want, free) — greedy, not a proportional share: a share would need
	// SHRINKING running leases when a newcomer arrives, and a lease is a
	// number already baked into a child's command line (`-j 8`), which no
	// message can take back. So no running lease is ever shrunk, and the
	// queue is never overtaken: a small request behind a large one waits,
	// because letting smalls pass would starve a large `min` for as long as
	// smalls keep coming. Nothing is granted unmeasured (`measured`).
	auto serveHead = [&] {
		while (measured() && !m_queue.isEmpty()) {
			const int free = freeCores();
			const CoreRequest& head = m_queue.first();
			if (free < head.min)
				break;
			const CoreRequest taken = head;
			m_queue.removeFirst();
			grant(taken, std::min(taken.want, free), false, free);
		}
	};
	serveHead();

	// Then the expired, wherever they stand: a waiter whose patience ran out
	// runs with its `min` now, over the budget, and says so (the grant is
	// marked forced and the badge turns red). The alternative — waiting on —
	// wedges a gate behind a neighbour's long run, and failing it turns load
	// into a red that means nothing; both are worse than one honestly
	// over-committed run. Granted even while unmeasured: a meter that never
	// produced a sample must not hold anyone forever either. After the head
	// pass, so a request that fits is never forced just because it asked for
	// no wait at all.
	bool expired = false;
	for (qsizetype i = 0; i < m_queue.size();) {
		if (m_queue[i].asked.msecsTo(now) >= m_queue[i].waitMs) {
			const CoreRequest taken = m_queue[i];
			m_queue.removeAt(i);
			grant(taken, taken.min, true, freeCores());
			expired = true;
		} else {
			++i;
		}
	}
	// An expired head leaving can unblock the one behind it only if that one
	// fits what is left; forced grants took cores, so usually it does not.
	if (expired)
		serveHead();
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
	return static_cast<double>(idleDelta) / static_cast<double>(totalDelta);
#else
	return -1.0;
#endif
}
