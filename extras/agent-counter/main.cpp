// ---------------------------------------------------------------------------
// AgentCounter — how many background agents are running, as a badge on screen.
//
// Started by `src/llm_agents/agent_counter.py` when no instance is listening; every
// agent then holds one open pipe connection for the whole of its turn, so the
// count is the number of live connections and needs neither pids nor polling.
// Whole design, including what was rejected: the beacon's module docstring.
//
// Modes:
//   (none)      become the counter, or exit quietly if one already runs
//   --status     print the count and the live agents (for a human or a script)
//   --status-json  print the structured status, including each agent's type
//   --quit       ask the running counter to exit (the badge cannot be clicked)
//   --pipe-name  print the name this build listens on, for the gate to compare
//   --selftest   the gate; no window
//   --badge-selftest [image.png]  offscreen UI checks and optional preview sheet
// ---------------------------------------------------------------------------
#include <QApplication>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLocalSocket>
#include <QStringList>
#include <QTimer>
#include <cstdio>

#include "Badge.h"
#include "BadgeTest.h"
#include "Registry.h"
#include "SelfTest.h"

#ifdef Q_OS_WIN
#include <windows.h>
#endif

namespace {

// How long the counter waits at zero before exiting. It is not a performance
// knob: a beacon that finds nobody starts a new counter in milliseconds, so
// the only thing this buys is not churning a process between two agents that
// follow each other closely.
constexpr int kIdleQuitMs = 20000;

void borrowParentConsole() {
#ifdef Q_OS_WIN
	// A Qt GUI executable has no console of its own, so `--status` run from a
	// terminal would print into nothing. Borrow the parent's — but ONLY when
	// this process has no usable stdout already: with `--status > file` the
	// handle is valid and reopening it on CONOUT$ would defeat the redirect.
	const HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
	if (out != nullptr && out != INVALID_HANDLE_VALUE)
		return;
	// ERROR_ACCESS_DENIED from AttachConsole means "this process is ALREADY
	// attached to a console" — which with a null stdout is precisely the case
	// that still needs CONOUT$ reopened. Returning on it printed nowhere.
	if (!AttachConsole(ATTACH_PARENT_PROCESS) && GetLastError() != ERROR_ACCESS_DENIED)
		return;
	FILE* reopened = nullptr;
	freopen_s(&reopened, "CONOUT$", "w", stdout);
	freopen_s(&reopened, "CONOUT$", "w", stderr);
#endif
}

QString humanAge(int seconds) {
	if (seconds < 60)
		return QStringLiteral("%1s").arg(seconds);
	return QStringLiteral("%1m%2s").arg(seconds / 60).arg(seconds % 60, 2, 10, QChar('0'));
}

// `--status` / `--quit`: connect, say one thing, leave. Never starts a counter —
// asking whether anything runs must not make something run.
int runClient(bool wantStatus, bool jsonStatus = false) {
	borrowParentConsole();
	QLocalSocket socket;
	socket.connectToServer(agentPipeName());
	if (!socket.waitForConnected(500)) {
		if (wantStatus)
			printf("%s\n", jsonStatus ? "{\"count\":0,\"agents\":[]}" : "0");
		return 0;
	}
	socket.write(wantStatus ? "{\"op\":\"status\"}\n" : "{\"op\":\"quit\"}\n");
	socket.flush();
	if (!wantStatus) {
		socket.waitForBytesWritten(500);
		return 0;
	}
	QByteArray reply;
	while (!reply.contains('\n') && socket.waitForReadyRead(1000))
		reply += socket.readAll();
	const QJsonObject answer = QJsonDocument::fromJson(reply).object();
	if (jsonStatus) {
		// A missing/truncated response is not evidence of zero running agents.
		if (!reply.contains('\n') || !answer.value(QStringLiteral("count")).isDouble()
		    || !answer.value(QStringLiteral("agents")).isArray()) {
			fprintf(stderr, "AgentCounter: incomplete or invalid status reply\n");
			return 1;
		}
		printf("%s\n", QJsonDocument(answer).toJson(QJsonDocument::Compact).constData());
		return 0;
	}
	// First line is the bare count so a script can read it with `head -1`; the
	// rest is for a human wondering which agent is the slow one.
	printf("%d\n", answer.value(QStringLiteral("count")).toInt());
	for (const QJsonValue& value : answer.value(QStringLiteral("agents")).toArray()) {
		const QJsonObject agent = value.toObject();
		printf("%s  %s\n",
		       agent.value(QStringLiteral("label")).toString().toUtf8().constData(),
		       humanAge(agent.value(QStringLiteral("age_s")).toInt()).toUtf8().constData());
	}
	return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
	QStringList args;
	for (int i = 1; i < argc; ++i)
		args += QString::fromLocal8Bit(argv[i]);

	const int badgeTest = args.indexOf(QStringLiteral("--badge-selftest"));
	if (badgeTest >= 0) {
		QApplication app(argc, argv);
		borrowParentConsole();
		const QString output = args.value(badgeTest + 1);
		return runBadgeSelfTest(output.startsWith(QLatin1String("--")) ? QString() : output);
	}
	if (args.contains(QStringLiteral("--selftest"))) {
		QCoreApplication app(argc, argv);
		return runSelfTest();
	}
	if (args.contains(QStringLiteral("--pipe-name"))) {
		// Exists for one reader: the cross-check test of the Python client, which
		// otherwise could only grep this source for a literal and would still
		// not be checking the user-name folding the two sides must agree on.
		QCoreApplication app(argc, argv);
		borrowParentConsole();
		printf("%s\n", agentPipeName().toUtf8().constData());
		return 0;
	}
	if (args.contains(QStringLiteral("--status")) || args.contains(QStringLiteral("--status-json"))
	    || args.contains(QStringLiteral("--quit"))) {
		QCoreApplication app(argc, argv);
		const bool jsonStatus = args.contains(QStringLiteral("--status-json"));
		return runClient(jsonStatus || args.contains(QStringLiteral("--status")), jsonStatus);
	}

	QApplication app(argc, argv);
	// The badge hides itself whenever the count is zero, and a hidden last
	// window would otherwise be read as "the application is done".
	app.setQuitOnLastWindowClosed(false);

	Registry registry;
	if (!registry.start(agentPipeName()))
		return 0;  // another instance owns the name: it is the counter, we are not

	Badge badge;
	QTimer idle;
	idle.setSingleShot(true);
	idle.setInterval(kIdleQuitMs);

	QObject::connect(&registry, &Registry::changed, &app, [&] {
		badge.setState(registry.groups());
		if (registry.count() == 0)
			idle.start();
		else
			idle.stop();
	});
	QObject::connect(&idle, &QTimer::timeout, &app, [&] {
		if (registry.retireIfIdle())
			app.quit();
		else
			idle.start();
	});
	QObject::connect(&registry, &Registry::quitRequested, &app, &QApplication::quit);

	// Nothing has connected yet, and the beacon that started this process is
	// already trying to: without this the counter would sit forever after a
	// start that nobody followed up on.
	idle.start();
	return app.exec();
}
