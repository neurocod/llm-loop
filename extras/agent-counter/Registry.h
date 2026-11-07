// ---------------------------------------------------------------------------
// The registry: who is running right now, learned from the wire alone.
//
// Transport lifecycle is pinned by `AgentCounter.exe --selftest`, independently
// of the translucent click-through window. Its rendering checks live in BadgeTest.
// ---------------------------------------------------------------------------
#pragma once

#include <QByteArray>
#include <QDateTime>
#include <QHash>
#include <QLocalServer>
#include <QObject>
#include <QString>
#include <QTimer>
#include <QVector>

#include <optional>

class QJsonValue;
class QLocalSocket;

// One live agent, as the counter knows it.
struct AgentEntry {
	QString label;
	QDateTime started;
	QString agentType = QStringLiteral("unknown");
	// What is LEFT of the provider's weekly limit, in whole percent, as this
	// agent's latest hello said; empty when that hello carried no figure.
	std::optional<int> weekLeft;
	QDateTime weekLeftAt;  // when the hello carrying `weekLeft` arrived
};

struct AgentGroup {
	QString agentType;
	int count = 0;
	int oldestSeconds = 0;
	// The weekly limit is the ACCOUNT's, so one figure per provider: the
	// freshest any of its agents reported. Empty = the badge's plain look.
	std::optional<int> weekLeft;
};

// Explicit wire types only: labels remain opaque, including legacy labels.
// Empty groups are omitted; display order is Claude, Codex, then unknown.
QVector<AgentGroup> groupAgents(const QVector<AgentEntry>& entries, const QDateTime& now);

// The hello's optional `week_left`: a JSON number in [0, 100], floored to a
// whole percent (never shows more left than there is). Anything else — absent,
// null, a string, out of range — is NO figure rather than a clamped one: a
// sender that says 140 % is broken, and the badge must not dress that up.
std::optional<int> weekLeftFrom(const QJsonValue& value);

// The pipe name, and the one rule about it: it is spelled a SECOND time in
// `src/llm_agents/agent_counter.py`, and a disagreement fails silently — the
// beacon finds no server, starts one, and still cannot reach it, so nothing is
// ever counted and nothing ever says why. A cross-check test reads both files
// and refuses a mismatch; that test is the only reason this may be a literal here.
QString agentPipeName();

class Registry : public QObject {
	Q_OBJECT
public:
	explicit Registry(QObject* parent = nullptr);
	~Registry() override;

	// Take the name, or report that another instance already holds it. This IS
	// the "is one already running" check — no window search, no lock file.
	//
	// It is NOT `QLocalServer::listen` alone, and the selftest is why: on
	// Windows a second listen on a live name SUCCEEDS. Qt creates pipes with
	// PIPE_UNLIMITED_INSTANCES, so the name is not an exclusive resource and
	// two counters would quietly share the incoming connections, each showing
	// its own half of the truth. See `claimName` for what makes it exclusive.
	bool start(const QString& name);

	int count() const;
	QVector<AgentEntry> entries() const;
	QVector<AgentGroup> groups() const;

	// Stop listening and answer whether the process may exit now.
	//
	// Closes the server FIRST, so the name is free the moment the decision is
	// made: a beacon connecting after this gets "not found", starts a fresh
	// counter and is counted there. Deciding while still listening would leave a
	// window in which a client connects to a server that is about to exit,
	// writes its hello into a dying pipe and is never counted at all — for the
	// whole turn, which is the one failure this program exists to prevent.
	bool retireIfIdle();

	// The reaper's patience, lowered by the selftest. A connection that says
	// nothing is not a registration: without this it would pin the process alive
	// forever and block `retireIfIdle`.
	void setHelloTimeoutMs(int ms) { m_helloTimeoutMs = ms; }

signals:
	void changed();
	void quitRequested();

private:
	// The idle-refusal test must observe server acceptance, not merely the
	// client's OS connection: on Windows those are separate events. Keep this
	// synchronization hook out of the public registry API.
	friend int runSelfTest();

	struct Client {
		QByteArray buffer;
		QDateTime connected;
		bool registered = false;
		AgentEntry entry;
	};

	// The exclusive half of `start`, and the only reason two counters cannot
	// exist: a pipe of our own created with FILE_FLAG_FIRST_PIPE_INSTANCE, which
	// no second process can create while this one holds it. Nothing ever
	// connects to it — it is a claim, not a channel — and it dies with the
	// process, so a killed counter leaves nothing behind to clean up.
	bool claimName(const QString& name);
	void releaseClaim();

	void onNewConnection();
	void onSocketClosed(QLocalSocket* socket);
	// Refuse a client outright — the answer to every malformed or silent one.
	// Both halves are needed at every site: `abort()` breaks the pipe, and the
	// explicit close is what takes the entry out of `m_clients`. Leaving that to
	// the `disconnected` signal would keep the client counted until Qt got round
	// to delivering it; arriving there twice is harmless, missing it is not.
	void dropClient(QLocalSocket* socket);
	void readFrom(QLocalSocket* socket);
	void handleLine(QLocalSocket* socket, const QByteArray& line);
	void sendStatus(QLocalSocket* socket);
	void onTick();
	void updateTicker();

	QLocalServer m_server;
	QString m_name;
	QHash<QLocalSocket*, Client> m_clients;
	QTimer m_tick;
	int m_helloTimeoutMs = 2000;
	void* m_claim = nullptr;  // HANDLE, kept opaque so windows.h stays out of here
};
