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
#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>

#include "Registry.h"

#ifdef Q_OS_WIN
#include <windows.h>
#endif

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

// One line of a conversation that may carry several (a lease's "queued" and
// later its "grant"): what follows the first newline stays in `pending` for
// the next call, which `readReply` would have thrown away.
QJsonObject readLine(QLocalSocket* socket, QByteArray& pending, const char* what) {
	check(waitFor([&] {
		pending += socket->readAll();
		return pending.contains('\n');
	}), what);
	const int cut = pending.indexOf('\n');
	if (cut < 0)
		return {};
	const QByteArray line = pending.left(cut);
	pending.remove(0, cut + 1);
	return QJsonDocument::fromJson(line).object();
}

CoreRequest coreRequest(quint64 id, int want, int min, const QDateTime& asked,
                        qint64 waitMs = 60'000) {
	CoreRequest request;
	request.id = id;
	request.label = QStringLiteral("r%1").arg(id);
	request.want = want;
	request.min = min;
	request.asked = asked;
	request.waitMs = waitMs;
	return request;
}

// The broker's rules on made-up machines: no clock, no meter, no wire.
void brokerRules() {
	const QDateTime t0 = QDateTime::fromSecsSinceEpoch(1'700'000'000);
	{
		CoreBroker broker(16);
		broker.request(coreRequest(1, 8, 2, t0));
		check(broker.settle(t0).isEmpty() && broker.queue().size() == 1,
		      "an unmeasured broker grants nothing: there is no load to subtract yet");
		broker.addIdleSample(1.0);
		QVector<CoreGrant> grants = broker.settle(t0);
		check(grants.size() == 1 && grants[0].id == 1 && grants[0].cores == 8 && !grants[0].forced,
		      "an idle machine grants the whole want");
		broker.request(coreRequest(2, 16, 2, t0));
		grants = broker.settle(t0);
		check(grants.size() == 1 && grants[0].cores == 8 && broker.freeCores() == 0,
		      "total - leased bounds an idle meter: a just-granted child is not burning yet");
		broker.request(coreRequest(3, 4, 2, t0));
		check(broker.settle(t0).isEmpty() && broker.queue().size() == 1,
		      "below its min a request waits");
		broker.release(1);
		grants = broker.settle(t0);
		check(grants.size() == 1 && grants[0].id == 3 && grants[0].cores == 4,
		      "a returned lease serves the queue");
		check(broker.leases().size() == 2 && broker.leases()[0].id == 2
		          && broker.leases()[0].cores == 8,
		      "a running lease is neither shrunk nor grown by others coming and going");
		broker.release(3);
		broker.release(2);
		check(broker.leased() == 0 && broker.leases().isEmpty(), "release takes the lease away");
	}
	{
		CoreBroker broker(16);
		broker.addIdleSample(0.5);
		check(broker.freeCores() == 8 && broker.foreignCores() == 8.0,
		      "unregistered load is subtracted from the free cores");
		broker.request(coreRequest(1, 12, 2, t0));
		const QVector<CoreGrant> grants = broker.settle(t0);
		check(grants.size() == 1 && grants[0].cores == 8 && grants[0].freeBefore == 8,
		      "and a grant gets only what the meter saw idle");
	}
	{
		CoreBroker broker(10);
		for (int i = 0; i < CoreBroker::kWindowSamples; ++i)
			broker.addIdleSample(0.0);
		check(broker.freeCores() == 0, "a busy window frees nothing");
		for (int i = 0; i < CoreBroker::kWindowSamples - 1; ++i)
			broker.addIdleSample(1.0);
		check(broker.freeCores() == 8, "the window is a mean of its samples");
		broker.addIdleSample(1.0);
		check(broker.freeCores() == 10, "and a sample past the window pushes the oldest out");
		broker.addIdleSample(1.7);
		broker.addIdleSample(-0.5);
		check(broker.idleCores() <= 10.0 && broker.freeCores() == 8,
		      "a meter outside [0, 1] is clamped, never minting cores");
	}
	{
		CoreBroker broker(16);
		broker.addIdleSample(1.0);
		broker.request(coreRequest(1, 13, 13, t0));
		broker.settle(t0);
		broker.request(coreRequest(2, 8, 6, t0));
		broker.request(coreRequest(3, 1, 1, t0));
		check(broker.settle(t0).isEmpty() && broker.queue().size() == 2,
		      "a small request does not overtake a waiting large one");
	}
	{
		CoreBroker broker(4);
		broker.addIdleSample(1.0);
		broker.request(coreRequest(1, 4, 4, t0));
		broker.settle(t0);
		broker.request(coreRequest(2, 3, 2, t0, 1000));
		check(broker.settle(t0.addMSecs(999)).isEmpty(), "a waiter inside its patience waits");
		const QVector<CoreGrant> grants = broker.settle(t0.addMSecs(1000));
		check(grants.size() == 1 && grants[0].cores == 2 && grants[0].forced
		          && grants[0].waitedMs == 1000 && broker.leased() == 6,
		      "past its patience it runs with its min, forced, over the machine");
	}
	{
		// The handoff: a 16-core run burning all of it ends, and the waiter
		// behind it must get the machine at once. The window still reads
		// the machine busy for 5 s; what it stores is the FOREIGN load of
		// those seconds (busy minus the lease's own use), which was nil.
		CoreBroker broker(16);
		broker.addIdleSample(1.0);
		broker.request(coreRequest(1, 16, 1, t0));
		broker.settle(t0);
		for (int i = 0; i < CoreBroker::kWindowSamples; ++i)
			broker.addSample(0.0, 16.0);
		broker.request(coreRequest(2, 16, 1, t0));
		check(broker.settle(t0).isEmpty(), "behind a busy whole-machine lease a request waits");
		broker.release(1);
		const QVector<CoreGrant> grants = broker.settle(t0);
		check(grants.size() == 1 && grants[0].id == 2 && grants[0].cores == 16,
		      "a released lease's cores are free at once, not after the meter's window");
	}
	{
		// The under-used lease: 8 held, 1 burning, 4 cores of foreign load.
		// The meter sees 11 idle, but 7 of them are the lease's to burn.
		CoreBroker broker(16);
		broker.addIdleSample(1.0);
		broker.request(coreRequest(1, 8, 8, t0));
		broker.settle(t0);
		for (int i = 0; i < CoreBroker::kWindowSamples; ++i)
			broker.addSample(11.0 / 16.0, 1.0);
		check(broker.freeCores() == 4 && std::abs(broker.foreignCores() - 4.0) < 1e-9,
		      "a lease is reserved whole: free = total - leased - foreign, not the idle cores");
		broker.request(coreRequest(2, 8, 1, t0));
		const QVector<CoreGrant> grants = broker.settle(t0);
		check(grants.size() == 1 && grants[0].cores == 4 && !grants[0].forced,
		      "so an idle-looking lease is not granted out from under it");
	}
	{
		// Only the head is decided: a follower that asked for no patience at
		// all, and would fit, still waits behind a head that waits.
		CoreBroker broker(16);
		broker.addIdleSample(1.0);
		broker.request(coreRequest(1, 13, 13, t0));
		broker.settle(t0);
		broker.request(coreRequest(2, 8, 6, t0, 1000));
		broker.request(coreRequest(3, 1, 1, t0, 0));
		check(broker.settle(t0).isEmpty() && broker.queue().size() == 2,
		      "an expired follower is not forced past a waiting head (FIFO holds under expiry)");
		check(broker.settle(t0.addMSecs(999)).isEmpty(),
		      "and free cores stay idle while the head waits: no backfill");
		const QVector<CoreGrant> grants = broker.settle(t0.addMSecs(1000));
		check(grants.size() == 2 && grants[0].id == 2 && grants[0].forced && grants[0].cores == 6
		          && grants[1].id == 3 && grants[1].forced && grants[1].cores == 1,
		      "the head's own deadline forces it, then the next head is decided");
	}
	{
		CoreBroker broker(16);
		broker.addIdleSample(1.0);
		broker.request(coreRequest(1, 12, 12, t0));
		broker.settle(t0);
		broker.request(coreRequest(2, 8, 6, t0, 5000));
		broker.request(coreRequest(3, 2, 2, t0, 1000));
		check(broker.settle(t0.addMSecs(1000)).isEmpty(),
		      "a follower's earlier deadline does not reorder the queue");
		broker.release(1);
		const QVector<CoreGrant> grants = broker.settle(t0.addMSecs(1000));
		check(grants.size() == 2 && grants[0].id == 2 && grants[0].cores == 8 && !grants[0].forced
		          && grants[1].id == 3 && grants[1].cores == 2 && !grants[1].forced,
		      "the head served first when cores return, then the one behind it if it fits");
	}
	{
		CoreBroker broker(4);
		broker.request(coreRequest(1, 2, 1, t0, 0));
		const QVector<CoreGrant> grants = broker.settle(t0);
		check(grants.size() == 1 && grants[0].forced && grants[0].cores == 1,
		      "even an unmeasured broker lets patience run out");
		CoreBroker measured(4);
		measured.addIdleSample(1.0);
		measured.request(coreRequest(1, 2, 1, t0, 0));
		const QVector<CoreGrant> fits = measured.settle(t0);
		check(fits.size() == 1 && !fits[0].forced && fits[0].cores == 2,
		      "a request that fits is served, never forced for asking no wait");
	}
	{
		// The budget (`/testSlow`): a ceiling on leases, not a smaller machine.
		CoreBroker broker(16);
		broker.addIdleSample(0.5);
		broker.setBudget(2);
		check(broker.total() == 2 && broker.machine() == 16 && broker.freeCores() == 2,
		      "a budget caps the free cores, while foreign load is still measured on the machine");
		broker.request(coreRequest(1, 8, 1, t0));
		QVector<CoreGrant> grants = broker.settle(t0);
		check(grants.size() == 1 && grants[0].cores == 2 && broker.freeCores() == 0,
		      "a budget cuts a grant to what it leaves");
		broker.setBudget(1);
		broker.request(coreRequest(2, 4, 1, t0));
		check(broker.settle(t0).isEmpty() && broker.leased() == 2 && broker.leases()[0].cores == 2,
		      "a budget below the running leases takes nothing back and grants nothing new");
		broker.setBudget(0);
		grants = broker.settle(t0);
		check(broker.total() == 16 && grants.size() == 1 && grants[0].cores == 4 && !grants[0].forced,
		      "lifting the budget serves the waiter from the whole machine again");
		broker.setBudget(40);
		check(broker.total() == 16, "a budget above the machine is the machine");
	}
	{
		// A request queued before the budget: its min is judged against the
		// budget when it is decided, so it fits instead of running forced.
		CoreBroker broker(16);
		broker.addIdleSample(1.0);
		broker.request(coreRequest(1, 12, 12, t0));
		broker.settle(t0);
		broker.request(coreRequest(2, 8, 6, t0, 5000));
		broker.setBudget(4);
		check(broker.settle(t0).isEmpty(), "under a budget the leased 12 keep the waiter out");
		broker.release(1);
		const QVector<CoreGrant> grants = broker.settle(t0);
		check(grants.size() == 1 && grants[0].cores == 4 && !grants[0].forced,
		      "a min above a budget set while it waited is clamped to the budget, not forced");
	}
	{
		CpuMeter meter;
		check(meter.sample() < 0.0, "the meter's first sample has no interval: no figure");
		pumpFor(200);
		const double idle = meter.sample();
		check(idle >= 0.0 && idle <= 1.0, "the real meter reads an idle fraction in [0, 1]");
		check(logicalCoreCount() >= 1, "the machine has at least one core");
	}
}

}  // namespace

// Leases over the pipe, on a registry with a made-up 8-core machine. Outside
// the anonymous namespace because Registry befriends it by name.
void selfTestLeases(const QString& base) {
	const QString name = base + QStringLiteral(".cores");
	Registry registry;
	registry.useFakeMeter(8);
	registry.setHelloTimeoutMs(300);
	check(registry.start(name), "the lease test takes its own isolated name");
	registry.m_broker.addIdleSample(1.0);

	QByteArray firstLines, secondLines;
	auto first = connectClient(name);
	say(first.get(), R"({"op":"lease","want":6,"min":2,"label":"suite"})");
	QJsonObject answer = readLine(first.get(), firstLines, "a lease is answered");
	check(answer.value(QStringLiteral("op")).toString() == QLatin1String("grant")
	          && answer.value(QStringLiteral("cores")).toInt() == 6
	          && answer.value(QStringLiteral("total")).toInt() == 8
	          && !answer.value(QStringLiteral("forced")).toBool(),
	      "the grant names its cores and the machine");
	check(registry.count() == 0, "a lease is not an agent");

	auto second = connectClient(name);
	say(second.get(), R"({"op":"lease","want":4,"min":3,"label":"waiter","wait_s":60})");
	answer = readLine(second.get(), secondLines, "a waiting lease is answered at once");
	check(answer.value(QStringLiteral("op")).toString() == QLatin1String("queued")
	          && answer.value(QStringLiteral("ahead")).toInt() == 0
	          && answer.value(QStringLiteral("free")).toInt() == 2,
	      "below its min it is told it waits, and why");

	first->abort();
	answer = readLine(second.get(), secondLines, "the waiter hears again");
	check(answer.value(QStringLiteral("op")).toString() == QLatin1String("grant")
	          && answer.value(QStringLiteral("cores")).toInt() == 4,
	      "a client killed mid-lease returns its cores, and the waiter is served");

	// Past the 300 ms hello deadline AND two of the reaper's 1 s ticks
	// (Registry.cpp, kTickMs): the deadline alone is checked only on a tick.
	pumpFor(2200);
	check(second->state() == QLocalSocket::ConnectedState && registry.cores().leased == 4,
	      "a lease outlives the hello reaper's patience");

	auto agent = connectClient(name);
	say(agent.get(), R"({"op":"hello","label":"agent","agent_type":"claude"})");
	checkCount(registry, 1, "an agent registers beside a lease");
	check(registry.cores().leased == 4, "and its hello holds no cores");

	auto probe = connectClient(name);
	say(probe.get(), R"({"op":"status"})");
	const QJsonObject status = readReply(probe.get(), "status answers with leases");
	const QJsonObject cores = status.value(QStringLiteral("cores")).toObject();
	const QJsonArray leases = cores.value(QStringLiteral("leases")).toArray();
	check(status.value(QStringLiteral("count")).toInt() == 1
	          && cores.value(QStringLiteral("total")).toInt() == 8
	          && cores.value(QStringLiteral("leased")).toInt() == 4 && leases.size() == 1
	          && leases.first().toObject().value(QStringLiteral("label")).toString()
	                 == QLatin1String("waiter"),
	      "status lists the leases beside the agents");

	say(second.get(), R"({"op":"lease","want":2})");
	answer = readLine(second.get(), secondLines, "a second lease on one connection is answered");
	check(answer.value(QStringLiteral("error")).toString() == QLatin1String("lease-held")
	          && registry.cores().leased == 4 && second->state() == QLocalSocket::ConnectedState,
	      "one lease per connection, and asking twice keeps the first");
	say(second.get(), R"({"op":"frobnicate"})");
	answer = readLine(second.get(), secondLines, "an unknown op on a lease is answered");
	pumpFor(200);
	check(answer.value(QStringLiteral("error")).toString() == QLatin1String("unknown-op")
	          && registry.cores().leased == 4 && second->state() == QLocalSocket::ConnectedState,
	      "an unknown op on a leasing connection is refused, and the running lease kept");
	say(second.get(), "not json");
	answer = readLine(second.get(), secondLines, "a garbage line on a lease is answered");
	pumpFor(200);
	check(answer.value(QStringLiteral("error")).toString() == QLatin1String("bad-json")
	          && registry.cores().leased == 4 && second->state() == QLocalSocket::ConnectedState,
	      "a garbage line on a leasing connection costs that line, not the lease");

	{
		QByteArray lines;
		auto split = connectClient(name);
		split->write(R"({"op":"lease","wa)");
		split->flush();
		pumpFor(150);
		say(split.get(), R"(nt":2,"label":"split"})");
		answer = readLine(split.get(), lines, "a lease in two writes is answered");
		check(answer.value(QStringLiteral("op")).toString() == QLatin1String("grant")
		          && answer.value(QStringLiteral("cores")).toInt() == 2,
		      "a lease line arriving in fragments is read whole");
		split->abort();
		check(waitFor([&] { return registry.cores().leased == 4; }), "and the fragment's lease returns");
	}
	{
		QByteArray aLines, bLines;
		auto a = connectClient(name);
		auto b = connectClient(name);
		say(a.get(), R"({"op":"lease","want":2,"min":2,"label":"a"})");
		say(b.get(), R"({"op":"lease","want":2,"min":2,"label":"b"})");
		const QJsonObject first = readLine(a.get(), aLines, "one of two simultaneous leases is answered");
		const QJsonObject other = readLine(b.get(), bLines, "the other is answered");
		check(first.value(QStringLiteral("op")).toString() == QLatin1String("grant")
		          && other.value(QStringLiteral("op")).toString() == QLatin1String("grant")
		          && registry.cores().leased == 8 && registry.m_broker.leases().size() == 3,
		      "two leases granted side by side fill the machine, each held on its own pipe");
		a->abort();
		check(waitFor([&] { return registry.cores().leased == 6; }),
		      "one holder dying returns its cores alone");
		b->abort();
		check(waitFor([&] { return registry.cores().leased == 4; }), "and then the other's");
	}
	{
		// The machine full (4 + 4); a waiter leaves before its turn.
		QByteArray blockLines, waitLines;
		auto blocker = connectClient(name);
		say(blocker.get(), R"({"op":"lease","want":4,"min":4,"label":"blocker"})");
		check(readLine(blocker.get(), blockLines, "the blocker is answered")
		              .value(QStringLiteral("cores")).toInt() == 4,
		      "the blocker fills the machine");
		auto waiter = connectClient(name);
		say(waiter.get(), R"({"op":"lease","want":2,"min":2,"label":"gone","wait_s":60})");
		check(readLine(waiter.get(), waitLines, "the waiter is answered")
		              .value(QStringLiteral("op")).toString() == QLatin1String("queued"),
		      "a waiter behind a full machine is queued");
		waiter->abort();
		check(waitFor([&] { return registry.m_broker.queue().isEmpty(); })
		          && registry.cores().leased == 8,
		      "a queued client that leaves takes its place in the queue with it");

		// Dying exactly as its grant is written: the blocker's return grants
		// `late`, whose pipe is already broken.
		auto late = connectClient(name);
		say(late.get(), R"({"op":"lease","want":2,"min":2,"label":"late","wait_s":60})");
		check(waitFor([&] { return registry.m_broker.queue().size() == 1; }), "the late one is queued");
		late->abort();
		blocker->abort();
		check(waitFor([&] {
			return registry.cores().leased == 4 && registry.m_broker.queue().isEmpty();
		}), "a client dying during its grant's delivery returns that grant");
	}
	{
		auto flood = connectClient(name);
		// A client that does not read: Qt would otherwise drain the pipe into
		// an unbounded buffer of its own, and the server never sees a backlog.
		flood->setReadBufferSize(1024);
		QByteArray lines;
		say(flood.get(), R"({"op":"lease","want":1,"label":"flood"})");
		readLine(flood.get(), lines, "the flooder's lease is answered");
		// Each line earns a "lease-held" reply it never reads.
		QByteArray burst;
		for (int i = 0; i < 4000; ++i)
			burst += R"({"op":"lease","want":1})" "\n";
		flood->write(burst);
		flood->flush();
		// Judged on the server's side: a client not reading does not notice
		// its pipe broke either.
		check(waitFor([&] { return registry.cores().leased == 4; }),
		      "a leaseholder piling up unread replies is dropped, and its lease returns");
	}
	{
		auto big = connectClient(name);
		QByteArray lines;
		say(big.get(), R"({"op":"lease","want":1,"label":"big"})");
		readLine(big.get(), lines, "the oversized one's lease is answered");
		big->write(QByteArray(70 * 1024, 'x'));
		big->flush();
		check(waitFor([&] {
			return big->state() != QLocalSocket::ConnectedState && registry.cores().leased == 4;
		}), "a line past the limit breaks the framing: even a leaseholder is dropped");
	}

	{
		QByteArray lines;
		auto newer = connectClient(name);
		say(newer.get(), R"({"op":"frobnicate"})");
		answer = readLine(newer.get(), lines, "an unknown op is answered");
		check(answer.value(QStringLiteral("op")).toString() == QLatin1String("error")
		          && answer.value(QStringLiteral("error")).toString() == QLatin1String("unknown-op")
		          && answer.value(QStringLiteral("got")).toString() == QLatin1String("frobnicate"),
		      "an unknown op is refused BY NAME, not cut off");
		check(waitFor([&] { return newer->state() != QLocalSocket::ConnectedState; }),
		      "and the refused connection is closed");
	}
	for (const QByteArray& bad : {QByteArray(R"({"op":"lease","want":2,"min":3})"),
	                              QByteArray(R"({"op":"lease","want":0})"),
	                              QByteArray(R"({"op":"lease","want":2.5})"),
	                              QByteArray(R"({"op":"lease"})"),
	                              QByteArray(R"({"op":"lease","want":2,"wait_s":"soon"})"),
	                              QByteArray(R"({"op":"lease","want":2,"job":5})"),
	                              QByteArray(R"({"op":"lease","want":2,"job":""})")}) {
		QByteArray lines;
		auto client = connectClient(name);
		say(client.get(), bad);
		answer = readLine(client.get(), lines, "a malformed lease is answered");
		check(answer.value(QStringLiteral("error")).toString() == QLatin1String("bad-lease")
		          && registry.cores().leased == 4 && registry.m_broker.queue().isEmpty(),
		      "a malformed lease is refused and takes nothing");
	}

	{
		QByteArray lines;
		auto hasty = connectClient(name);
		say(hasty.get(), R"({"op":"lease","want":50,"min":50,"label":"hasty","wait_s":0})");
		answer = readLine(hasty.get(), lines, "an impatient lease is answered");
		check(answer.value(QStringLiteral("op")).toString() == QLatin1String("grant")
		          && answer.value(QStringLiteral("forced")).toBool()
		          && answer.value(QStringLiteral("cores")).toInt() == 8,
		      "a min above the machine is clamped to it, and no patience means forced at once");
		check(registry.cores().leased == 12 && registry.cores().leased > registry.cores().total,
		      "the badge sees the over-commitment");
		hasty->abort();
		check(waitFor([&] { return registry.cores().leased == 4; }), "and its return");
	}
	{
		// The budget over the pipe: 4 leased on an 8-core machine, budget 2.
		QByteArray setLines, waitLines, statusLines, offLines;
		auto setter = connectClient(name);
		say(setter.get(), R"({"op":"budget","cores":2})");
		answer = readLine(setter.get(), setLines, "a budget is answered");
		check(answer.value(QStringLiteral("op")).toString() == QLatin1String("budget")
		          && answer.value(QStringLiteral("budget")).toInt() == 2
		          && answer.value(QStringLiteral("total")).toInt() == 2
		          && answer.value(QStringLiteral("machine")).toInt() == 8
		          && answer.value(QStringLiteral("leased")).toInt() == 4,
		      "the budget's answer names the ceiling, the machine and what is leased");
		check(waitFor([&] { return setter->state() != QLocalSocket::ConnectedState; })
		          && registry.cores().leased == 4 && registry.cores().total == 2,
		      "the budget connection closes, the running lease is kept, the badge shows the ceiling");
		auto waiter = connectClient(name);
		say(waiter.get(), R"({"op":"lease","want":2,"label":"under-budget","wait_s":60})");
		check(readLine(waiter.get(), waitLines, "a lease under a full budget is answered")
		              .value(QStringLiteral("op")).toString() == QLatin1String("queued"),
		      "with the budget held a new lease waits, though the machine has room");
		auto probe = connectClient(name);
		say(probe.get(), R"({"op":"budget"})");
		answer = readLine(probe.get(), statusLines, "a budget query is answered");
		check(answer.value(QStringLiteral("budget")).toInt() == 2 && registry.m_broker.budget() == 2,
		      "a budget op without cores only asks");
		auto off = connectClient(name);
		say(off.get(), R"({"op":"budget","cores":null})");
		answer = readLine(off.get(), offLines, "lifting the budget is answered");
		check(answer.value(QStringLiteral("budget")).isNull()
		          && answer.value(QStringLiteral("total")).toInt() == 8,
		      "null lifts the budget");
		answer = readLine(waiter.get(), waitLines, "the waiter hears once the budget is lifted");
		check(answer.value(QStringLiteral("op")).toString() == QLatin1String("grant")
		          && answer.value(QStringLiteral("cores")).toInt() == 2,
		      "a lifted budget serves the queue at once");
		waiter->abort();
		check(waitFor([&] { return registry.cores().leased == 4; }), "the budget waiter returns");
		for (const QByteArray& bad : {QByteArray(R"({"op":"budget","cores":0})"),
		                              QByteArray(R"({"op":"budget","cores":"two"})"),
		                              QByteArray(R"({"op":"budget","cores":1.5})")}) {
			QByteArray lines;
			auto client = connectClient(name);
			say(client.get(), bad);
			answer = readLine(client.get(), lines, "a malformed budget is answered");
			check(answer.value(QStringLiteral("error")).toString() == QLatin1String("bad-budget")
			          && registry.m_broker.budget() == 0,
			      "a malformed budget is refused and sets nothing");
		}
	}

#ifdef Q_OS_WIN
	{
		// A lease naming its job: the broker reads that job's CPU time as the
		// lease's own use. An empty job burns nothing, so on a machine fully
		// busy the 4 cores the job-less lease beside it does not account for
		// are foreign — where assuming the job's lease busy would hide them.
		const QString jobName = QStringLiteral("Local\\agent-counter-selftest-%1")
		                            .arg(QCoreApplication::applicationPid());
		const HANDLE job =
		    CreateJobObjectW(nullptr, reinterpret_cast<const wchar_t*>(jobName.utf16()));
		check(job != nullptr, "the selftest makes a job object");
		QByteArray lines, missingLines;
		auto jobbed = connectClient(name);
		say(jobbed.get(), QJsonDocument(QJsonObject{{QStringLiteral("op"), QStringLiteral("lease")},
		                                            {QStringLiteral("want"), 4},
		                                            {QStringLiteral("job"), jobName}})
		                      .toJson(QJsonDocument::Compact));
		answer = readLine(jobbed.get(), lines, "a lease with a job is answered");
		check(answer.value(QStringLiteral("cores")).toInt() == 4
		          && answer.value(QStringLiteral("measured")).toBool(),
		      "a lease naming its job is granted measured");
		// The leases released above left their last stretch for the next
		// step (the meter is fake here and never stepped); not this case's.
		registry.m_departedSeconds = 0.0;
		registry.meterStep(0.0, 1.0);
		check(std::abs(registry.m_broker.foreignCores() - 2.0) < 1e-9
		          && registry.m_broker.freeCores() == 0,
		      "the job's measured use, not the lease's size, is subtracted from the busy machine");
		jobbed->abort();
		check(waitFor([&] { return registry.cores().leased == 4; }), "the measured lease returns");
		CloseHandle(job);

		auto missing = connectClient(name);
		say(missing.get(), R"({"op":"lease","want":1,"job":"Local\\agent-counter-no-such-job"})");
		answer = readLine(missing.get(), missingLines, "a lease with a missing job is answered");
		check(answer.value(QStringLiteral("op")).toString() == QLatin1String("grant")
		          && !answer.value(QStringLiteral("measured")).toBool(true),
		      "a job that does not open leaves the lease granted but unmeasured, and says so");
		missing->abort();
		check(waitFor([&] { return registry.cores().leased == 4; }), "and it returns too");
		// The window back to an idle machine for what follows.
		for (int i = 0; i < CoreBroker::kWindowSamples; ++i)
			registry.m_broker.addIdleSample(1.0);
	}
#endif

	second->disconnectFromServer();
	agent->abort();
	check(waitFor([&] { return registry.idle(); }),
	      "with the last lease and agent gone the registry is idle");
	check(registry.cores().shown, "the cores block stays a moment after the last lease ends");
	QElapsedTimer hold;
	hold.start();
	check(waitFor([&] { return !registry.cores().shown; }) && hold.elapsed() >= kCoresHoldMs / 2,
	      "and then hides: no CPU block on an idle machine (kCoresHoldMs)");

	registry.m_broker.setBudget(2);
	check(!registry.idle(), "a set budget keeps the counter from retiring: it would be lost");
	registry.m_broker.setBudget(0);
	check(registry.idle(), "and a lifted one lets it go");
}

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

	brokerRules();
	selfTestLeases(name);

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
