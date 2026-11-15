#include "Registry.h"

#include <QCoreApplication>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLocalSocket>
#include <QPointer>

#include <algorithm>
#include <cmath>
#include <limits>

#ifdef Q_OS_WIN
#include <windows.h>
#endif

namespace {

// A line longer than this is not a hello from anything of ours. Bounded so a
// peer that never sends a newline cannot grow the buffer without limit.
constexpr int kMaxLineBytes = 64 * 1024;

// Replies written but not yet read by the client (Registry::reply). Ours read
// every line as it comes, and a whole conversation is a few hundred bytes, so
// a backlog this size means a client that writes and never reads.
constexpr qint64 kMaxUnsentReplyBytes = 64 * 1024;

// A job name on the wire is a kernel object name: Windows caps those at
// MAX_PATH, and nothing of ours names one longer than ~40 characters.
constexpr int kMaxJobNameChars = 260;

// Labels appear in the badge's `--status` listing, not on screen; a cap
// keeps one buggy caller from making that output unreadable.
constexpr int kMaxLabelChars = 120;

// While anything is registered the badge repaints once a second, because its
// colour encodes the oldest agent's AGE and would otherwise freeze at the
// colour it had when the last agent started. The same tick reaps connections
// that never said hello, so there is one timer, not two.
constexpr int kTickMs = 1000;

// The meter's sample interval: CoreBroker::kWindowSamples of these make its
// window. GetSystemTimes deltas are exact whatever the timer's jitter.
constexpr int kMeterMs = 1000;

// A lease request that names no `wait_s` waits this long for its `min`
// before it is granted `min` forced (CoreBroker::settle). The Python client
// always names one (`llm_agents.cores`, DEFAULT_WAIT_S, the same figure);
// the cap keeps a typo from parking a request for a year.
constexpr double kDefaultWaitS = 300.0;
constexpr double kMaxWaitS = 24 * 3600.0;

// A lease field that must be a whole number >= 1, or nullopt.
std::optional<int> positiveInt(const QJsonValue& value) {
	if (!value.isDouble())
		return std::nullopt;
	const double number = value.toDouble();
	if (!std::isfinite(number) || number < 1.0 || number != std::floor(number)
	    || number > 1e6)
		return std::nullopt;
	return static_cast<int>(number);
}

QString normalizedAgentType(const QString& type) {
	if (type == QLatin1String("claude") || type == QLatin1String("codex"))
		return type;
	return QStringLiteral("unknown");
}

// Only `[a-z0-9]` of the user name survives, ASCII-folded. Deliberately crude:
// `src/llm_agents/agent_counter.py` must reproduce this character for character.
// Anything richer (Qt's Unicode-aware toLower against Python's) could differ on a name neither
// author will ever type.
QString foldedUser() {
	const QString raw = qEnvironmentVariable("USERNAME");
	QString out;
	for (const QChar ch : raw) {
		const char16_t c = ch.unicode();
		if (c >= u'A' && c <= u'Z')
			out += QChar(c - u'A' + u'a');
		else if ((c >= u'a' && c <= u'z') || (c >= u'0' && c <= u'9'))
			out += ch;
	}
	return out.isEmpty() ? QStringLiteral("anon") : out.left(32);
}

}  // namespace

QVector<AgentGroup> groupAgents(const QVector<AgentEntry>& entries, const QDateTime& now) {
	QVector<AgentGroup> groups{
	    {QStringLiteral("claude")}, {QStringLiteral("codex")}, {QStringLiteral("unknown")}};
	QVector<QDateTime> figureAt(groups.size());
	for (const AgentEntry& entry : entries) {
		const QString type = normalizedAgentType(entry.agentType);
		for (qsizetype i = 0; i < groups.size(); ++i) {
			AgentGroup& group = groups[i];
			if (group.agentType != type)
				continue;
			++group.count;
			const qint64 age = std::clamp<qint64>(
			    entry.started.secsTo(now), 0, std::numeric_limits<int>::max());
			group.oldestSeconds = std::max(group.oldestSeconds, static_cast<int>(age));
			// Freshest, not lowest: the figure is one account's, so an older
			// report is simply out of date, not a second opinion. A tie keeps
			// the first, which only matters within one millisecond.
			if (entry.weekLeft && (!group.weekLeft || entry.weekLeftAt > figureAt[i])) {
				group.weekLeft = entry.weekLeft;
				figureAt[i] = entry.weekLeftAt;
			}
			break;
		}
	}
	groups.erase(std::remove_if(groups.begin(), groups.end(),
	                            [](const AgentGroup& group) { return group.count == 0; }),
	             groups.end());
	return groups;
}

std::optional<int> weekLeftFrom(const QJsonValue& value) {
	if (!value.isDouble())
		return std::nullopt;
	const double percent = value.toDouble();
	if (!std::isfinite(percent) || percent < 0.0 || percent > 100.0)
		return std::nullopt;
	return static_cast<int>(std::floor(percent));
}

QString agentPipeName() {
	// The version is part of the NAME, and that is the whole upgrade story: a
	// new client simply does not find an old running instance, starts its own,
	// and two protocol versions never half-understand each other.
	//
	// v2: the `lease` op and the first replies on a hello-only channel, and
	// unknown ops refused by name instead of a silent drop.
	return QStringLiteral("llm-agents.v2.") + foldedUser();
}

Registry::Registry(QObject* parent) : QObject(parent) {
	connect(&m_server, &QLocalServer::newConnection, this, &Registry::onNewConnection);
	m_tick.setInterval(kTickMs);
	connect(&m_tick, &QTimer::timeout, this, &Registry::onTick);
	m_meterTick.setInterval(kMeterMs);
	connect(&m_meterTick, &QTimer::timeout, this, &Registry::onMeterTick);
	m_clock.start();
	m_coresHold.setSingleShot(true);
	m_coresHold.setInterval(kCoresHoldMs);
	connect(&m_coresHold, &QTimer::timeout, this, &Registry::changed);
}

void Registry::useFakeMeter(int total) {
	m_fakeMeter = true;
	m_meterTick.stop();
	m_broker = CoreBroker(total);
}

Registry::~Registry() {
	// Close BEFORE releasing, the same order `retireIfIdle` uses and for the
	// same reason. A member `QLocalServer` is destroyed after this body, so
	// releasing first left a window in which a successor could claim the name
	// and listen while this instance was still listening too — and a beacon
	// routed to this one would write its hello into a server about to vanish.
	m_server.close();
	releaseClaim();
	for (const LeaseUse& use : std::as_const(m_leaseUse))
		closeJob(use.job);
}

bool Registry::claimName(const QString& name) {
#ifdef Q_OS_WIN
	const QString path = QStringLiteral("\\\\.\\pipe\\") + name + QStringLiteral(".claim");
	const HANDLE claim = CreateNamedPipeW(
	    reinterpret_cast<const wchar_t*>(path.utf16()),
	    PIPE_ACCESS_DUPLEX | FILE_FLAG_FIRST_PIPE_INSTANCE,
	    PIPE_TYPE_BYTE | PIPE_WAIT, 1, 0, 0, 0, nullptr);
	if (claim != INVALID_HANDLE_VALUE) {
		m_claim = claim;
		return true;
	}
	// Only "taken" means taken. Reading EVERY failure as a rival counter turns
	// an out-of-resources or bad-parameter failure into a silent exit with
	// nobody listening at all, and beacons then start losers forever; so
	// anything else runs unclaimed, on the grounds that a counter which might
	// briefly be one of two beats no counter.
	//
	// Two codes, not the one the documentation names. CreateNamedPipe is
	// documented to fail with ERROR_ACCESS_DENIED when FILE_FLAG_FIRST_PIPE_INSTANCE
	// meets an existing pipe — measured here, with nMaxInstances at 1, it is
	// ERROR_PIPE_BUSY (231): the instance limit is checked first. Trusting the
	// documented code alone made every second instance believe the name was
	// free, which is the exact failure the claim exists to prevent (and which
	// the selftest caught within one run of being introduced).
	const DWORD failure = GetLastError();
	return failure != ERROR_ACCESS_DENIED && failure != ERROR_PIPE_BUSY;
#else
	// Elsewhere the transport name IS exclusive (a bound AF_UNIX path), so
	// `listen` alone decides and there is nothing to claim.
	Q_UNUSED(name);
	return true;
#endif
}

void Registry::releaseClaim() {
#ifdef Q_OS_WIN
	if (m_claim != nullptr) {
		CloseHandle(static_cast<HANDLE>(m_claim));
		m_claim = nullptr;
	}
#endif
}

bool Registry::start(const QString& name) {
	m_name = name;
	if (!claimName(name))
		return false;
	// A DACL limited to this user. Named pipes are machine-wide, and the
	// default would let any local account inflate the count.
	//
	// Note what is NOT here: `QLocalServer::removeServer`. On Windows the pipe
	// dies with its process, so there is never a stale one to clear — and the
	// call would delete a LIVE instance's name.
	m_server.setSocketOptions(QLocalServer::UserAccessOption);
	if (m_server.listen(name)) {
		if (!m_fakeMeter) {
			m_meter.sample();  // the baseline the first interval is measured from
			m_meterTick.start();
		}
		return true;
	}
	releaseClaim();
	return false;
}

int Registry::count() const {
	int live = 0;
	for (const Client& client : m_clients)
		if (client.registered)
			++live;
	return live;
}

QVector<AgentEntry> Registry::entries() const {
	QVector<AgentEntry> out;
	for (const Client& client : m_clients)
		if (client.registered)
			out.append(client.entry);
	std::sort(out.begin(), out.end(), [](const AgentEntry& a, const AgentEntry& b) {
		return a.started < b.started;
	});
	return out;
}

QVector<AgentGroup> Registry::groups() const {
	return groupAgents(entries(), QDateTime::currentDateTime());
}

CoreSummary Registry::cores() const {
	CoreSummary summary;
	summary.leased = m_broker.leased();
	summary.total = m_broker.total();
	summary.queued = static_cast<int>(m_broker.queue().size());
	summary.shown = !m_broker.leases().isEmpty() || summary.queued > 0 || m_coresHold.isActive();
	return summary;
}

bool Registry::idle() const {
	return count() == 0 && m_broker.leases().isEmpty() && m_broker.queue().isEmpty();
}

bool Registry::retireIfIdle() {
	// A connection the OS has already completed may not have been dispatched
	// yet — the beacon believes it is connected and has written its hello,
	// while `m_clients` is still empty. Closing on that reading loses it, and
	// the beacon never retries: that turn goes uncounted for its whole life.
	// One pump delivers everything already completed. What remains is the
	// microseconds-wide window of a connect completing DURING the close, which
	// nothing short of an acknowledged registration can close — and that would
	// put a blocking read in the beacon's path, which is a worse trade for a
	// badge (see src/llm_agents/agent_counter.py on why nothing there may block).
	QCoreApplication::processEvents(QEventLoop::AllEvents);
	m_server.close();
	if (m_clients.isEmpty()) {
		// The claim goes with the name, and before this call returns: the
		// successor cannot start until it is gone, and the beacon waiting to be
		// counted is the one paying for every millisecond in between.
		releaseClaim();
		return true;
	}
	// Somebody connected in the gap and has not spoken yet. Keep serving —
	// unless the name was taken meanwhile, in which case that other instance is
	// the counter now and this one has nothing left to be.
	if (m_server.listen(m_name))
		return false;
	releaseClaim();
	return true;
}

void Registry::onNewConnection() {
	while (QLocalSocket* socket = m_server.nextPendingConnection()) {
		Client client;
		client.connected = QDateTime::currentDateTime();
		m_clients.insert(socket, client);
		connect(socket, &QLocalSocket::readyRead, this, [this, socket] { readFrom(socket); });
		connect(socket, &QLocalSocket::disconnected, this, [this, socket] { onSocketClosed(socket); });
		// An agent killed outright (`taskkill /F`, which codex_agent itself
		// uses) breaks the pipe rather than closing it politely, and Qt reports
		// that as an error, not always as `disconnected`. Both doors lead here,
		// because a registration that outlives its process is the one state
		// this counter must never show.
		connect(socket, &QLocalSocket::errorOccurred, this,
		        [this, socket] { onSocketClosed(socket); });
		updateTicker();
	}
}

void Registry::onSocketClosed(QLocalSocket* socket) {
	const auto it = m_clients.find(socket);
	if (it == m_clients.end())
		return;  // both signals fired; the first one already took it
	const bool counted = it->registered;
	const quint64 leaseId = it->leaseId;
	m_clients.erase(it);
	socket->deleteLater();
	updateTicker();
	// The return of a lease is this and nothing else: the kernel closing a
	// dead client's handle lands here exactly as a polite exit does.
	if (leaseId != 0)
		releaseLease(leaseId);
	if (counted || leaseId != 0)
		emit changed();
}

void Registry::releaseLease(quint64 leaseId) {
	m_leaseSockets.remove(leaseId);
	const auto use = m_leaseUse.find(leaseId);
	if (use != m_leaseUse.end()) {
		// Its stretch since the last meter step, by the wall clock (capped
		// at one interval: longer is the timer having stalled, and the
		// machine's next sample covers no more than one).
		const double stretchS =
		    std::clamp((m_clock.elapsed() - use->accountedMs) / 1000.0, 0.0, kMeterMs / 1000.0);
		for (const CoreLease& lease : m_broker.leases())
			if (lease.id == leaseId)
				m_departedSeconds += accountLease(leaseId, lease.cores, stretchS);
		// After the last reading: this handle may be the job's last, and a
		// kill-on-close job ends whatever the dead client left running.
		closeJob(use->job);
		m_leaseUse.erase(use);
	}
	m_broker.release(leaseId);
	settleLeases();
	if (m_broker.leases().isEmpty() && m_broker.queue().isEmpty())
		m_coresHold.start();
	else
		m_coresHold.stop();
}

double Registry::accountLease(quint64 id, int cores, double stretchS) {
	LeaseUse& use = m_leaseUse[id];
	use.accountedMs = m_clock.elapsed();
	const double cap = cores * stretchS;
	const double total = jobCpuSeconds(use.job);
	if (total < 0.0)
		return cap;  // no job: assumed to burn all it holds
	const double used = std::max(0.0, total - use.jobSeconds);
	use.jobSeconds = total;
	use.lastCores = stretchS > 0.0 ? used / stretchS : 0.0;
	return std::min(used, cap);
}

void Registry::dropClient(QLocalSocket* socket) {
	socket->abort();
	onSocketClosed(socket);
}

void Registry::readFrom(QLocalSocket* socket) {
	if (!m_clients.contains(socket))
		return;
	if (m_clients[socket].closing) {
		socket->readAll();
		return;
	}
	m_clients[socket].buffer += socket->readAll();
	// The entry is looked up afresh every round rather than held across the
	// loop: `handleLine` can close the connection and erase it, and the stale
	// iterator would then be read back.
	for (;;) {
		const auto it = m_clients.find(socket);
		if (it == m_clients.end() || it->closing)
			return;
		const int cut = it->buffer.indexOf('\n');
		if (cut < 0) {
			// The bound is on the unfinished LINE, checked after the whole
			// ones are taken: a burst of many short lines in one read is not
			// a line too long.
			if (it->buffer.size() > kMaxLineBytes)
				dropClient(socket);
			return;
		}
		const QByteArray line = it->buffer.left(cut);
		it->buffer.remove(0, cut + 1);
		handleLine(socket, line);
	}
}

void Registry::handleLine(QLocalSocket* socket, const QByteArray& line) {
	const QJsonDocument doc = QJsonDocument::fromJson(line.trimmed());
	if (!doc.isObject()) {
		// A leasing connection keeps its cores over one bad line (`refuse`);
		// anything else that sends garbage is not ours and goes at once.
		if (m_clients[socket].leaseId != 0)
			refuse(socket, QStringLiteral("bad-json"), {});
		else
			dropClient(socket);
		return;
	}
	const QJsonObject obj = doc.object();
	const QString op = obj.value(QStringLiteral("op")).toString();
	if (op == QLatin1String("hello")) {
		Client& client = m_clients[socket];
		QString label = obj.value(QStringLiteral("label")).toString().left(kMaxLabelChars);
		if (label.isEmpty())
			label = QStringLiteral("agent");
		client.entry.label = label;
		// Additive v1 field: old clients and unrecognized values still count,
		// but cannot acquire a provider identity from their human-readable label.
		client.entry.agentType = normalizedAgentType(
		    obj.value(QStringLiteral("agent_type")).toString());
		// Additive v1 field too, and the reason the weekly figure rides on a
		// repeated hello instead of an op of its own: a v1 counter dropped a
		// client on an unknown op (v2 refuses it by name and closes), so a
		// beacon sending a new op would lose its registration either way; an
		// unknown FIELD is ignored. Each hello is the sender's whole state, so
		// one without the field clears the figure.
		client.entry.weekLeft = weekLeftFrom(obj.value(QStringLiteral("week_left")));
		client.entry.weekLeftAt =
		    client.entry.weekLeft ? QDateTime::currentDateTime() : QDateTime();
		if (!client.registered) {
			client.registered = true;
			client.entry.started = QDateTime::currentDateTime();
		}
		emit changed();
	} else if (op == QLatin1String("status")) {
		// Answering does NOT register: a probe asking "how many" must not
		// become one of the things it counts.
		sendStatus(socket);
		socket->disconnectFromServer();
	} else if (op == QLatin1String("lease")) {
		handleLease(socket, obj);
	} else if (op == QLatin1String("quit")) {
		emit quitRequested();
	} else {
		// Refused BY NAME: a newer client asking for an op this build lacks
		// learns that it was this op, and can fall back instead of reading a
		// cut pipe as a crash. A v2 beacon never reads, so its pipe merely
		// holds a line nobody collects; nothing of the old behaviour is lost
		// but the silence.
		refuse(socket, QStringLiteral("unknown-op"), {{QStringLiteral("got"), op}});
	}
}

// The lease, on the wire:
//   -> {"op":"lease","want":8,"min":2,"label":"npm test","wait_s":300}
//   <- {"op":"queued","ahead":1,"free":1,"total":16}   only when it must wait
//   <- {"op":"grant","cores":8,"total":16,"free":11,"foreign":2.5,
//       "forced":false,"waited_s":0.0}
//       "forced":false,"waited_s":0.0,"measured":true}
// `want` is required, `min` defaults to 1, `wait_s` to kDefaultWaitS; both
// counts are clamped to the machine (a min above it could never be served).
// `job` (optional) names the client's job object holding the leased child:
// the broker reads that job's CPU time to tell the lease's own load from
// foreign load (`meterStep`); `measured` in the grant says whether it opened.
// The lease lasts as long as the connection: there is no "release" op, so
// no client can forget one.
void Registry::handleLease(QLocalSocket* socket, const QJsonObject& obj) {
	if (m_clients[socket].leaseId != 0) {
		// Answered, and the connection kept (`refuse`): it holds the first.
		refuse(socket, QStringLiteral("lease-held"), {});
		return;
	}
	const std::optional<int> want = positiveInt(obj.value(QStringLiteral("want")));
	const QJsonValue minValue = obj.value(QStringLiteral("min"));
	const std::optional<int> min =
	    minValue.isUndefined() ? std::optional<int>(1) : positiveInt(minValue);
	const QJsonValue waitValue = obj.value(QStringLiteral("wait_s"));
	const double waitS = waitValue.isUndefined() ? kDefaultWaitS : waitValue.toDouble(-1.0);
	const QJsonValue jobValue = obj.value(QStringLiteral("job"));
	if (!want || !min || *min > *want || (!waitValue.isUndefined() && !waitValue.isDouble())
	    || !std::isfinite(waitS) || waitS < 0.0
	    || (!jobValue.isUndefined()
	        && (!jobValue.isString() || jobValue.toString().isEmpty()
	            || jobValue.toString().size() > kMaxJobNameChars))) {
		refuse(socket, QStringLiteral("bad-lease"),
		       {{QStringLiteral("detail"),
		         QStringLiteral("want and min are whole numbers >= 1 with min <= want; "
		                        "wait_s is a number >= 0; job is a non-empty object name")}});
		return;
	}
	Client& client = m_clients[socket];
	CoreRequest request;
	request.id = m_nextLeaseId++;
	request.label = obj.value(QStringLiteral("label")).toString().left(kMaxLabelChars);
	if (request.label.isEmpty())
		request.label = QStringLiteral("lease");
	request.want = std::min(*want, m_broker.total());
	request.min = std::min(*min, m_broker.total());
	request.asked = QDateTime::currentDateTime();
	request.waitMs = static_cast<qint64>(std::min(waitS, kMaxWaitS) * 1000.0);
	client.leaseId = request.id;
	m_leaseSockets.insert(request.id, socket);
	LeaseUse use;
	use.job = openJob(jobValue.toString());
	m_leaseUse.insert(request.id, use);
	m_broker.request(request);
	const int free = m_broker.freeCores();
	settleLeases();
	for (qsizetype i = 0; i < m_broker.queue().size(); ++i) {
		if (m_broker.queue()[i].id != request.id)
			continue;
		reply(socket, {{QStringLiteral("op"), QStringLiteral("queued")},
		               {QStringLiteral("ahead"), static_cast<int>(i)},
		               {QStringLiteral("free"), free},
		               {QStringLiteral("total"), m_broker.total()}});
		break;
	}
	emit changed();
}

void Registry::reply(QLocalSocket* socket, const QJsonObject& message) {
	const auto it = m_clients.find(socket);
	if (it == m_clients.end() || it->closing)
		return;
	socket->write(QJsonDocument(message).toJson(QJsonDocument::Compact) + '\n');
	socket->flush();
	if (socket->bytesToWrite() <= kMaxUnsentReplyBytes)
		return;
	it->closing = true;  // read and answered no more from here on
	const QPointer<QLocalSocket> guard(socket);
	QMetaObject::invokeMethod(this, [this, guard] {
		if (guard && m_clients.contains(guard.data()))
			dropClient(guard.data());
	}, Qt::QueuedConnection);
}

void Registry::refuse(QLocalSocket* socket, const QString& error, const QJsonObject& detail) {
	const auto it = m_clients.find(socket);
	if (it == m_clients.end())
		return;
	QJsonObject message = detail;
	message.insert(QStringLiteral("op"), QStringLiteral("error"));
	message.insert(QStringLiteral("error"), error);
	const bool leasing = it->leaseId != 0;
	reply(socket, message);  // `it` stays valid: a drop from `reply` is queued
	if (leasing)
		return;
	// Out of every count NOW, closed once the reply is written: the
	// `disconnected` signal then erases the entry (onSocketClosed).
	const bool counted = it->registered;
	it->registered = false;
	it->closing = true;
	socket->disconnectFromServer();
	if (counted)
		emit changed();
}

void Registry::settleLeases() {
	const QVector<CoreGrant> grants = m_broker.settle(QDateTime::currentDateTime());
	const qint64 nowMs = m_clock.elapsed();
	for (const CoreGrant& grant : grants) {
		LeaseUse& use = m_leaseUse[grant.id];
		use.granted = true;
		use.accountedMs = nowMs;
		use.jobSeconds = std::max(0.0, jobCpuSeconds(use.job));
		QLocalSocket* socket = m_leaseSockets.value(grant.id);
		if (socket == nullptr)
			continue;
		reply(socket, {{QStringLiteral("op"), QStringLiteral("grant")},
		               {QStringLiteral("cores"), grant.cores},
		               {QStringLiteral("total"), m_broker.total()},
		               {QStringLiteral("free"), grant.freeBefore},
		               {QStringLiteral("foreign"), std::round(m_broker.foreignCores() * 10) / 10},
		               {QStringLiteral("forced"), grant.forced},
		               {QStringLiteral("waited_s"), grant.waitedMs / 1000.0},
		               {QStringLiteral("measured"), use.job != nullptr}});
	}
	if (!grants.isEmpty())
		emit changed();
}

QJsonObject Registry::coreStatus() const {
	const QDateTime now = QDateTime::currentDateTime();
	QJsonArray leases;
	for (const CoreLease& lease : m_broker.leases()) {
		const LeaseUse use = m_leaseUse.value(lease.id);
		QJsonObject entry{{QStringLiteral("label"), lease.label},
		                  {QStringLiteral("cores"), lease.cores},
		                  {QStringLiteral("forced"), lease.forced},
		                  {QStringLiteral("measured"), use.job != nullptr},
		                  {QStringLiteral("age_s"),
		                   static_cast<double>(lease.granted.secsTo(now))}};
		// Present only once measured: a lease without a job has no figure,
		// and 0 would read as "idle".
		if (use.lastCores >= 0.0)
			entry.insert(QStringLiteral("used"), std::round(use.lastCores * 10) / 10);
		leases.append(entry);
	}
	QJsonArray queued;
	for (const CoreRequest& request : m_broker.queue())
		queued.append(QJsonObject{{QStringLiteral("label"), request.label},
		                          {QStringLiteral("want"), request.want},
		                          {QStringLiteral("min"), request.min},
		                          {QStringLiteral("age_s"),
		                           static_cast<double>(request.asked.secsTo(now))}});
	return QJsonObject{
	    {QStringLiteral("total"), m_broker.total()},
	    {QStringLiteral("leased"), m_broker.leased()},
	    {QStringLiteral("free"), m_broker.freeCores()},
	    {QStringLiteral("measured"), m_broker.measured()},
	    {QStringLiteral("idle"), std::round(m_broker.idleCores() * 10) / 10},
	    {QStringLiteral("foreign"), std::round(m_broker.foreignCores() * 10) / 10},
	    {QStringLiteral("leases"), leases},
	    {QStringLiteral("queued"), queued},
	};
}

void Registry::sendStatus(QLocalSocket* socket) {
	const QDateTime now = QDateTime::currentDateTime();
	QJsonArray agents;
	for (const AgentEntry& entry : entries()) {
		QJsonObject agent{
		    {QStringLiteral("label"), entry.label},
		    {QStringLiteral("agent_type"), entry.agentType},
		    {QStringLiteral("age_s"), static_cast<double>(entry.started.secsTo(now))},
		};
		// Present only when known, as on the wire.
		if (entry.weekLeft)
			agent.insert(QStringLiteral("week_left"), *entry.weekLeft);
		agents.append(agent);
	}
	reply(socket, {
	    {QStringLiteral("count"), count()},
	    {QStringLiteral("agents"), agents},
	    // Additive: v1 readers of `count`/`agents` see what they always saw.
	    {QStringLiteral("cores"), coreStatus()},
	});
}

void Registry::onTick() {
	const QDateTime now = QDateTime::currentDateTime();
	const QList<QLocalSocket*> sockets = m_clients.keys();
	for (QLocalSocket* socket : sockets) {
		const auto it = m_clients.find(socket);
		if (it == m_clients.end() || it->registered || it->leaseId != 0)
			continue;
		if (it->connected.msecsTo(now) >= m_helloTimeoutMs)
			dropClient(socket);
	}
	if (count() > 0)
		emit changed();
}

void Registry::onMeterTick() {
	const double idle = m_meter.sample();
	if (idle >= 0.0) {
		meterStep(idle, m_meter.intervalSeconds());
		return;
	}
	// Every second, not only on arrivals and departures: free cores also
	// appear when foreign load ends, and a waiter's patience runs out on the
	// clock (CoreBroker::settle).
	settleLeases();
}

void Registry::meterStep(double idleFraction, double intervalS) {
	if (!(intervalS > 0.0)) {
		m_broker.addIdleSample(idleFraction);
		settleLeases();
		return;
	}
	// A live lease is credited over the whole interval, even one granted
	// half-way through it: a lease without a job is then assumed busy for a
	// half second it was not, which over-states its own use (and so
	// under-states foreign load) by at most its size for one sample of five
	// — the price of not keeping a second clock beside the meter's.
	double ownSeconds = m_departedSeconds;
	m_departedSeconds = 0.0;
	for (const CoreLease& lease : m_broker.leases())
		ownSeconds += accountLease(lease.id, lease.cores, intervalS);
	const double own = ownSeconds / intervalS;
	m_broker.addSample(idleFraction, own);
	settleLeases();
}

void Registry::updateTicker() {
	if (m_clients.isEmpty())
		m_tick.stop();
	else if (!m_tick.isActive())
		m_tick.start();
}
