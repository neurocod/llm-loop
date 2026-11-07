// ---------------------------------------------------------------------------
// The gate for everything in this program that can be wrong.
//
// Runs against its OWN pipe name (the live counter's name with a suffix), so a
// gate run while agents are working neither steals their connections nor
// reports their count as its own.
// ---------------------------------------------------------------------------
#include "SelfTest.h"

#include <QCoreApplication>
#include <QElapsedTimer>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLocalSocket>
#include <cstdio>
#include <functional>
#include <memory>

#include "Registry.h"

namespace {

int g_failures = 0;

// Whole native gate measured 2.272 s on 2026-09-08. This also raises the old
// 3 s wait by 10 s, leaving slow machines room to deliver local pipe events.
constexpr int kWaitMs = 13000;

void check(bool ok, const char* what) {
	printf("%s  %s\n", ok ? "PASS" : "FAIL", what);
	if (!ok)
		++g_failures;
}

// Pump the event loop until `done` or the deadline. Every assertion in this
// file is about something the loop delivers (a connection, a line, a broken
// pipe), so a bare check would race the very machinery it tests.
bool waitFor(const std::function<bool()>& done, int ms = kWaitMs) {
	QElapsedTimer clock;
	clock.start();
	while (!done() && clock.elapsed() < ms)
		QCoreApplication::processEvents(QEventLoop::WaitForMoreEvents, 20);
	return done();
}

// Run the loop for a fixed spell, for the assertions that something did NOT
// happen. `waitFor` is the wrong tool there and silently so: a predicate that is
// already true returns before a single event is dispatched, so the check passes
// against a server that never even ran. That is how "connecting alone does not
// count" used to be tested.
void pumpFor(int ms) {
	QElapsedTimer clock;
	clock.start();
	while (clock.elapsed() < ms)
		QCoreApplication::processEvents(QEventLoop::WaitForMoreEvents, 10);
}

// The commonest assertion in this file, and never without the wait: every
// change to the count crosses the wire, so a bare read races the delivery it is
// checking. Written out eight times, it was also eight chances to forget that.
void checkCount(const Registry& registry, int expected, const char* what) {
	check(waitFor([&] { return registry.count() == expected; }), what);
}

// One line of answer, accumulated rather than read once: `readAll` hands over
// whatever has arrived so far, and a single call can land mid-line.
QJsonObject readReply(QLocalSocket* socket, const char* what) {
	QByteArray reply;
	check(waitFor([&] {
		reply += socket->readAll();
		return reply.contains('\n');
	}), what);
	return QJsonDocument::fromJson(reply).object();
}

std::unique_ptr<QLocalSocket> connectClient(const QString& name) {
	auto socket = std::make_unique<QLocalSocket>();
	socket->connectToServer(name);
	socket->waitForConnected(kWaitMs);
	return socket;
}

void say(QLocalSocket* socket, const QByteArray& line) {
	socket->write(line + '\n');
	socket->flush();
}

}  // namespace

int runSelfTest() {
	const QString name = agentPipeName() + QStringLiteral(".selftest");

	Registry registry;
	check(registry.start(name), "the first instance takes the name");

	{
		Registry rival;
		check(!rival.start(name), "a second instance loses the name (this IS the "
		                          "already-running check)");
	}

	{
		auto client = connectClient(name);
		check(client->state() == QLocalSocket::ConnectedState, "a client connects");
		pumpFor(200);
		check(registry.count() == 0,
		      "connecting alone does not count: only a hello does");
		say(client.get(), R"({"op":"hello","label":"review"})");
		checkCount(registry, 1, "hello counts an agent");
		check(!registry.entries().isEmpty()
		          && registry.entries().first().label == QLatin1String("review"),
		      "the label survives the wire");
		check(!registry.entries().isEmpty()
		          && registry.entries().first().agentType == QLatin1String("unknown"),
		      "a legacy hello without an agent type remains unknown");

		// The case the whole design rests on: no goodbye message, no pid, the
		// process simply dies. `abort` breaks the pipe exactly as a killed
		// process does.
		client->abort();
		checkCount(registry, 0,
		           "an agent that dies without saying goodbye stops being counted");
	}

	{
		auto first = connectClient(name);
		auto second = connectClient(name);
		say(first.get(), R"({"op":"hello","label":"a","agent_type":"claude"})");
		say(second.get(), R"({"op":"hello","label":"b","agent_type":"codex"})");
		checkCount(registry, 2, "two agents count as two");
		const QVector<AgentGroup> both = registry.groups();
		check(both.size() == 2 && both[0].agentType == QLatin1String("claude")
		          && both[0].count == 1 && both[1].agentType == QLatin1String("codex")
		          && both[1].count == 1,
		      "explicit provider types form separate ordered groups");

		auto probe = connectClient(name);
		say(probe.get(), R"({"op":"status"})");
		const QJsonObject answer = readReply(probe.get(), "status answers");
		check(answer.value(QStringLiteral("count")).toInt() == 2, "status reports the count");
		check(answer.value(QStringLiteral("agents")).toArray().size() == 2,
		      "status lists the agents");
		QHash<QString, QString> reportedTypes;
		for (const QJsonValue& agent : answer.value(QStringLiteral("agents")).toArray()) {
			const QJsonObject item = agent.toObject();
			reportedTypes.insert(item.value(QStringLiteral("label")).toString(),
			                     item.value(QStringLiteral("agent_type")).toString());
		}
		check(reportedTypes.value(QStringLiteral("a")) == QLatin1String("claude")
		          && reportedTypes.value(QStringLiteral("b")) == QLatin1String("codex"),
		      "status echoes each explicit provider type");
		check(registry.count() == 2, "asking how many does not make the asker one of them");

		second->disconnectFromServer();
		checkCount(registry, 1, "a clean exit decrements too");
		check(registry.groups().size() == 1
		          && registry.groups().first().agentType == QLatin1String("claude"),
		      "disconnect removes only the departing provider group");
		const QDateTime originalStart = registry.entries().value(0).started;
		check(waitFor([&] { return QDateTime::currentDateTime() > originalStart; }),
		      "the clock advances before testing that re-hello preserves age");
		say(first.get(), R"({"op":"hello","label":"a-new","agent_type":"codex"})");
		check(waitFor([&] {
			return registry.entries().value(0).label == QLatin1String("a-new");
		}), "a repeated hello updates the existing registration");
		check(registry.count() == 1 && registry.groups().size() == 1
		          && registry.groups().first().agentType == QLatin1String("codex")
		          && registry.entries().value(0).started == originalStart,
		      "changing provider on re-hello neither double-counts nor resets the age");
		first->abort();
		checkCount(registry, 0, "and so does the last one");
	}

	{
		// Each payload is otherwise a valid hello. A misleading label must not
		// turn a missing, future, or malformed type into a known provider.
		const QVector<QByteArray> hellos{
		    R"({"op":"hello","label":"claude:legacy"})",
		    R"({"op":"hello","label":"codex:unknown","agent_type":"unknown"})",
		    R"({"op":"hello","label":"claude:future","agent_type":"other"})",
		    R"({"op":"hello","label":"codex:malformed","agent_type":42})",
		    R"({"op":"hello","label":"claude:null","agent_type":null})",
		    R"({"op":"hello","label":"codex:empty","agent_type":""})"};
		for (const QByteArray& hello : hellos) {
			auto client = connectClient(name);
			say(client.get(), hello);
			checkCount(registry, 1, "a hello with an unspecified provider still counts");
			check(registry.entries().value(0).agentType == QLatin1String("unknown")
			          && registry.groups().size() == 1
			          && registry.groups().first().agentType == QLatin1String("unknown"),
			      "unrecognized provider values stay unknown regardless of the label");
			auto probe = connectClient(name);
			say(probe.get(), R"({"op":"status"})");
			const QJsonObject answer = readReply(probe.get(), "unknown provider status answers");
			const QJsonArray agents = answer.value(QStringLiteral("agents")).toArray();
			check(!agents.isEmpty() && agents.first().toObject()
			              .value(QStringLiteral("agent_type")).toString() == QLatin1String("unknown"),
			      "status echoes the normalized unknown provider");
			client->abort();
			checkCount(registry, 0, "the unknown provider disconnects normally");
		}
	}

	{
		// The weekly-limit figure rides on the hello (Registry::handleLine on why).
		auto client = connectClient(name);
		say(client.get(), R"({"op":"hello","label":"week","agent_type":"claude"})");
		checkCount(registry, 1, "a hello without week_left counts as before");
		check(!registry.groups().value(0).weekLeft,
		      "without week_left the group has no figure: the badge's old look");
		say(client.get(), R"({"op":"hello","label":"week","agent_type":"claude","week_left":17})");
		check(waitFor([&] { return registry.groups().value(0).weekLeft == 17; }),
		      "a re-hello's week_left reaches the provider group");
		check(registry.count() == 1, "and the figure update is not a second agent");
		auto probe = connectClient(name);
		say(probe.get(), R"({"op":"status"})");
		const QJsonObject answer = readReply(probe.get(), "status answers with a figure");
		const QJsonObject agent =
		    answer.value(QStringLiteral("agents")).toArray().first().toObject();
		check(agent.value(QStringLiteral("week_left")).toInt(-1) == 17,
		      "status echoes week_left");
		say(client.get(), R"({"op":"hello","label":"week","agent_type":"claude"})");
		check(waitFor([&] { return !registry.groups().value(0).weekLeft; }),
		      "a hello without the field clears the figure: each hello is the whole state");
		say(client.get(), R"({"op":"hello","label":"week","agent_type":"claude","week_left":24.9})");
		check(waitFor([&] { return registry.groups().value(0).weekLeft == 24; }),
		      "a fractional remainder is floored, never rounded up");
		client->abort();
		checkCount(registry, 0, "the figure's agent leaves normally");

		check(!weekLeftFrom(QJsonValue(-1)) && !weekLeftFrom(QJsonValue(100.5))
		          && !weekLeftFrom(QJsonValue(QStringLiteral("17")))
		          && !weekLeftFrom(QJsonValue(QJsonValue::Null))
		          && !weekLeftFrom(QJsonValue(QJsonValue::Undefined)),
		      "out-of-range, string, null and absent week_left are no figure");
		check(weekLeftFrom(QJsonValue(0)) == 0 && weekLeftFrom(QJsonValue(100)) == 100,
		      "the range ends 0 and 100 are figures");

		const QDateTime now = QDateTime::fromSecsSinceEpoch(1'700'000'000);
		AgentEntry older{QStringLiteral("older"), now.addSecs(-300), QStringLiteral("codex")};
		older.weekLeft = 12;
		older.weekLeftAt = now.addSecs(-10);
		AgentEntry newer{QStringLiteral("newer"), now.addSecs(-5), QStringLiteral("codex")};
		newer.weekLeft = 40;
		newer.weekLeftAt = now.addSecs(-1);
		AgentEntry silent{QStringLiteral("silent"), now.addSecs(-1), QStringLiteral("codex")};
		AgentEntry other{QStringLiteral("other"), now, QStringLiteral("claude")};
		other.weekLeft = 90;
		other.weekLeftAt = now;
		const QVector<AgentGroup> byFreshness = groupAgents({older, silent, newer, other}, now);
		check(byFreshness.size() == 2 && byFreshness[0].weekLeft == 90
		          && byFreshness[1].count == 3 && byFreshness[1].weekLeft == 40,
		      "one figure per provider: the freshest report, not the lowest, and a "
		      "silent agent does not erase it");
	}

	{
		const QDateTime now = QDateTime::fromSecsSinceEpoch(1'700'000'000);
		const QVector<AgentEntry> entries{
		    {QStringLiteral("future"), now.addSecs(60), QStringLiteral("unknown")},
		    {QStringLiteral("new-codex"), now.addSecs(-5), QStringLiteral("codex")},
		    {QStringLiteral("old-claude"), now.addSecs(-120), QStringLiteral("claude")},
		    {QStringLiteral("old-codex"), now.addSecs(-30), QStringLiteral("codex")}};
		const QVector<AgentGroup> groups = groupAgents(entries, now);
		check(groups.size() == 3 && groups[0].agentType == QLatin1String("claude")
		          && groups[0].count == 1 && groups[0].oldestSeconds == 120
		          && groups[1].agentType == QLatin1String("codex")
		          && groups[1].count == 2 && groups[1].oldestSeconds == 30
		          && groups[2].agentType == QLatin1String("unknown")
		          && groups[2].count == 1 && groups[2].oldestSeconds == 0,
		      "group ages are independent, ordered, and clamp future timestamps to zero");
		check(groupAgents({}, now).isEmpty(), "empty groups are omitted entirely");
	}

	{
		// Two lines in ONE write, because that is where the read loop can trip:
		// handling the first can erase the very entry the loop is holding (the
		// status reply closes the connection), and a single-line-per-write test
		// never reaches that path.
		auto client = connectClient(name);
		client->write("{\"op\":\"hello\",\"label\":\"pair\"}\n{\"op\":\"status\"}\n");
		client->flush();
		const QJsonObject answer =
		    readReply(client.get(), "two lines arriving in one read are both handled");
		check(answer.value(QStringLiteral("count")).toInt() == 1,
		      "and in order: the hello counted before the status answered");
		client->abort();
		checkCount(registry, 0, "and the pair leaves cleanly");
	}

	{
		// Several at once: the reaper walks the same container it erases from.
		// Its accelerated deadline belongs only to this registry: using it for
		// the whole gate can reap the silent client in the retirement test.
		Registry reaper;
		reaper.setHelloTimeoutMs(300);
		const QString reaperName = name + QStringLiteral(".reaper");
		check(reaper.start(reaperName), "the reaper test takes its own isolated name");
		auto first = connectClient(reaperName);
		auto second = connectClient(reaperName);
		auto third = connectClient(reaperName);
		check(waitFor([&] {
			return first->state() != QLocalSocket::ConnectedState
			    && second->state() != QLocalSocket::ConnectedState
			    && third->state() != QLocalSocket::ConnectedState;
		}), "connections that never say hello are all reaped");
	}

	{
		// The refusal path of `retireIfIdle`, which decides the counter's whole
		// lifetime and was never exercised: it must keep serving, not half-exit.
		//
		// QLocalSocket::waitForConnected on Windows only checks client state;
		// it does not dispatch the server's newConnection notification. Wait
		// for an accepted, unregistered socket before testing this branch.
		check(waitFor([&] { return registry.m_clients.isEmpty(); }),
		      "earlier connections have drained before the silent-client case");
		auto pending = connectClient(name);
		check(waitFor([&] { return !registry.m_clients.isEmpty(); }),
		      "the server accepts the silent client before the retirement check");
		check(!registry.retireIfIdle(), "a connected but silent client refuses retirement");
		auto after = connectClient(name);
		check(after->state() == QLocalSocket::ConnectedState,
		      "and the name is still served after the refusal");
		say(after.get(), R"({"op":"hello","label":"post-refusal"})");
		checkCount(registry, 1, "a client arriving after the refusal is counted normally");
		after->abort();
		pending->abort();
		checkCount(registry, 0, "and both go away");
	}

	{
		auto rude = connectClient(name);
		say(rude.get(), "this is not json");
		check(waitFor([&] { return rude->state() != QLocalSocket::ConnectedState; }),
		      "a garbage line closes the connection");
		check(registry.count() == 0, "and counts nothing");
	}

	check(registry.retireIfIdle(), "an idle registry agrees to retire");
	{
		Registry successor;
		check(successor.start(name),
		      "retiring frees the name before the process goes, so the next "
		      "beacon starts a counter it can actually reach");
	}

	if (g_failures == 0)
		printf("\nselftest: all cases passed\n");
	else
		printf("\nselftest: %d case(s) FAILED\n", g_failures);
	return g_failures == 0 ? 0 : 1;
}
