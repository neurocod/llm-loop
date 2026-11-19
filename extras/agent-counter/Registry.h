// ---------------------------------------------------------------------------
// The registry: who is running right now, learned from the wire alone.
//
// Transport lifecycle is pinned by `AgentCounter.exe --selftest`, independently
// of the translucent click-through window. Its rendering checks live in BadgeTest.
// ---------------------------------------------------------------------------
#pragma once

#include <QByteArray>
#include <QDateTime>
#include <QElapsedTimer>
#include <QHash>
#include <QLocalServer>
#include <QObject>
#include <QString>
#include <QTimer>
#include <QVector>

#include <optional>

#include "CoreBroker.h"

class QJsonObject;
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

// What the badge shows of the core broker: leased / total, plus how many wait.
// `shown` while anything is leased or queued, and for kCoresHoldMs after the
// last lease ends — an agent's hello holds no cores, so the plain agent badge
// looks exactly as it did before leases, and a badge with nothing else left
// hides whole.
struct CoreSummary {
	bool shown = false;
	int leased = 0;
	int total = 0;
	int queued = 0;
};

// How long the cores block stays on the badge after the last lease ended and
// the queue emptied: long enough to see the run finish ("0/16"), short enough
// that an idle machine shows no CPU block — user's decision 2026-11-15, after
// a "CPU 0/16" badge was left on screen with nothing running.
constexpr int kCoresHoldMs = 1000;

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
// ever counted and nothing ever says why. The cross-check
// `test_the_pipe_prefix_is_spelled_the_same_in_both_halves` (tests/
// test_agent_counter.py, needs no build) reads both files and refuses a
// mismatch; that test is the only reason this may be a literal here.
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
	CoreSummary cores() const;
	// Nothing to serve: no agent counted, no lease held, nobody queued, no
	// budget set.
	bool idle() const;

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
	// The lease half of the gate (SelfTest.cpp): it fakes the machine
	// (`useFakeMeter`) and feeds the broker's samples itself.
	friend void selfTestLeases(const QString& base);

	struct Client {
		QByteArray buffer;
		QDateTime connected;
		bool registered = false;
		AgentEntry entry;
		// Nonzero once the client asked for cores (queued or granted). A
		// lease is a connection of its own kind: it is not an agent (it does
		// not raise `count`), and it is exempt from the hello reaper.
		quint64 leaseId = 0;
		// Refused by name and closing: the reply is on its way out, so the
		// connection is no longer read, counted or leased.
		bool closing = false;
	};

	// How a lease's own CPU use is measured (`meterStep`). `job` is the
	// client's job object, opened by the name its lease carried; null = no
	// job (none named, or it would not open) and the lease is assumed to
	// burn everything it holds (CoreBroker::addIdleSample on what that costs).
	struct LeaseUse {
		void* job = nullptr;
		bool granted = false;
		double jobSeconds = 0.0;  // the job's CPU total at the last reading
		qint64 accountedMs = 0;   // m_clock at the last reading
		double lastCores = -1.0;  // measured use over the last interval; < 0 = none yet
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
	void handleLease(QLocalSocket* socket, const QJsonObject& obj);
	void handleBudget(QLocalSocket* socket, const QJsonObject& obj);
	// The broker's budget as the wire spells it: a number, or null for none.
	QJsonValue budgetValue() const;
	// Answer `{"op":"error","error":<error>,...}` and close once it is
	// written — the reply to an op this build does not know or a lease it
	// cannot take. Not `dropClient`: a client that asked something and was
	// cut off without a word cannot tell "refused" from "crashed", and a
	// newer client meeting this build must be able to say WHICH.
	//
	// Except on a connection holding a lease (queued or granted): its reply
	// is sent and the connection KEPT, because closing it would return the
	// cores of a child that is running right now. Each line stands alone, so
	// a stray one (an op this build lacks, a second lease, garbage) costs that
	// line only. What does end a leasing connection is what breaks the
	// channel itself: a line past kMaxLineBytes (no framing left to resync
	// on) and replies it never reads (`reply`).
	void refuse(QLocalSocket* socket, const QString& error, const QJsonObject& detail);
	// Write one line. A client that lets kMaxUnsentReplyBytes pile up unread
	// is dropped — a lease client is exempt from the hello reaper, so this
	// is the only bound on what it can make this process buffer; its lease
	// returns through the ordinary close. The drop is QUEUED: callers are
	// mid-way through the client's entry or the broker's grants.
	void reply(QLocalSocket* socket, const QJsonObject& message);
	// Run the broker's decisions and tell each granted client its cores.
	void settleLeases();
	// Return a lease (queued or granted): its last stretch of use is
	// accounted, its job let go, the queue served.
	void releaseLease(quint64 leaseId);
	// The CPU seconds lease `id` used since its last reading, `stretchS` of
	// wall time ago, capped at its size times that stretch (use BEYOND the
	// lease is foreign load: nothing reserved it); without a job, the cap
	// itself. Advances the reading.
	double accountLease(quint64 id, int cores, double stretchS);
	void sendStatus(QLocalSocket* socket);
	QJsonObject coreStatus() const;
	void onTick();
	void onMeterTick();
	// One meter interval of `intervalS` seconds at `idleFraction`: the
	// leases' own use is measured and the sample handed to the broker, then
	// the queue is settled. The selftest calls it with a made-up machine.
	void meterStep(double idleFraction, double intervalS);
	void updateTicker();
	// The selftest's machine: `total` cores and no real meter, so samples
	// come only from the test (`m_broker.addIdleSample`).
	void useFakeMeter(int total);

	QLocalServer m_server;
	QString m_name;
	QHash<QLocalSocket*, Client> m_clients;
	QTimer m_tick;
	int m_helloTimeoutMs = 2000;
	CoreBroker m_broker{logicalCoreCount()};
	CpuMeter m_meter;
	// Always running while the process lives, unlike `m_tick`: the meter's
	// window must already hold samples when the first lease arrives, and a
	// sample a second costs one system call.
	QTimer m_meterTick;
	bool m_fakeMeter = false;
	quint64 m_nextLeaseId = 1;
	QHash<quint64, QLocalSocket*> m_leaseSockets;
	QHash<quint64, LeaseUse> m_leaseUse;
	// Running for kCoresHoldMs after the last lease and waiter left; its
	// timeout repaints the badge without the cores block.
	QTimer m_coresHold;
	// Own CPU seconds of leases released since the last meter step: the
	// interval they ended in still carries their load in the machine's busy
	// figure, and without this it would read as foreign.
	double m_departedSeconds = 0.0;
	QElapsedTimer m_clock;
	void* m_claim = nullptr;  // HANDLE, kept opaque so windows.h stays out of here
};
