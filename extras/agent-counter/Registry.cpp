#include "Registry.h"

#include <QCoreApplication>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLocalSocket>

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

// Labels appear in the badge's `--status` listing, not on screen; a cap
// keeps one buggy caller from making that output unreadable.
constexpr int kMaxLabelChars = 120;

// While anything is registered the badge repaints once a second, because its
// colour encodes the oldest agent's AGE and would otherwise freeze at the
// colour it had when the last agent started. The same tick reaps connections
// that never said hello, so there is one timer, not two.
constexpr int kTickMs = 1000;

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
	return QStringLiteral("llm-agents.v1.") + foldedUser();
}

Registry::Registry(QObject* parent) : QObject(parent) {
	connect(&m_server, &QLocalServer::newConnection, this, &Registry::onNewConnection);
	m_tick.setInterval(kTickMs);
	connect(&m_tick, &QTimer::timeout, this, &Registry::onTick);
}

Registry::~Registry() {
	// Close BEFORE releasing, the same order `retireIfIdle` uses and for the
	// same reason. A member `QLocalServer` is destroyed after this body, so
	// releasing first left a window in which a successor could claim the name
	// and listen while this instance was still listening too — and a beacon
	// routed to this one would write its hello into a server about to vanish.
	m_server.close();
	releaseClaim();
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
	if (m_server.listen(name))
		return true;
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
	m_clients.erase(it);
	socket->deleteLater();
	updateTicker();
	if (counted)
		emit changed();
}

void Registry::dropClient(QLocalSocket* socket) {
	socket->abort();
	onSocketClosed(socket);
}

void Registry::readFrom(QLocalSocket* socket) {
	if (!m_clients.contains(socket))
		return;
	m_clients[socket].buffer += socket->readAll();
	if (m_clients[socket].buffer.size() > kMaxLineBytes) {
		dropClient(socket);
		return;
	}
	// The entry is looked up afresh every round rather than held across the
	// loop: `handleLine` can close the connection and erase it, and the stale
	// iterator would then be read back.
	for (;;) {
		const auto it = m_clients.find(socket);
		if (it == m_clients.end())
			return;
		const int cut = it->buffer.indexOf('\n');
		if (cut < 0)
			return;
		const QByteArray line = it->buffer.left(cut);
		it->buffer.remove(0, cut + 1);
		handleLine(socket, line);
	}
}

void Registry::handleLine(QLocalSocket* socket, const QByteArray& line) {
	const QJsonDocument doc = QJsonDocument::fromJson(line.trimmed());
	if (!doc.isObject()) {
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
		// repeated hello instead of an op of its own: a counter built before it
		// drops a client on an unknown op, so a new beacon talking to an old
		// running badge would lose its registration; an unknown FIELD is
		// ignored. Each hello is the sender's whole state, so one without the
		// field clears the figure.
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
	} else if (op == QLatin1String("quit")) {
		emit quitRequested();
	} else {
		dropClient(socket);
	}
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
	const QJsonObject reply{
	    {QStringLiteral("count"), count()},
	    {QStringLiteral("agents"), agents},
	};
	socket->write(QJsonDocument(reply).toJson(QJsonDocument::Compact) + '\n');
	socket->flush();
}

void Registry::onTick() {
	const QDateTime now = QDateTime::currentDateTime();
	const QList<QLocalSocket*> sockets = m_clients.keys();
	for (QLocalSocket* socket : sockets) {
		const auto it = m_clients.find(socket);
		if (it == m_clients.end() || it->registered)
			continue;
		if (it->connected.msecsTo(now) >= m_helloTimeoutMs)
			dropClient(socket);
	}
	if (count() > 0)
		emit changed();
}

void Registry::updateTicker() {
	if (m_clients.isEmpty())
		m_tick.stop();
	else if (!m_tick.isActive())
		m_tick.start();
}
