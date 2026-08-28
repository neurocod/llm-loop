// PreToolUse gate, C++ port of ../hooks/ask_user_gate.py.
//
// The Python script stays the NORMATIVE reference: it carries the full
// rationale for every rule (why each shape defeats the allow-list rather than
// merely missing it, why the list is deliberately short, why the chain finding
// needs a `cd` to survive). Read it first; the comments here cover only what is
// specific to the port. This file must produce byte-identical verdicts and
// byte-identical refusal text -- except the `Blocked by <path>` line, which
// names whichever copy ran -- and `cpp/parity_check.py` is the gate that says
// so: it feeds one corpus to the reference `scan()`/`render()` and to this
// binary and diffs the output.
//
// Why a port at all: the hook runs before EVERY shell call. Measured on this
// machine, the interpreter start alone is ~70 ms and this binary is ~7 ms; at
// the user's ~100k calls a month that is roughly two hours of wall clock a
// month spent starting Python. Nothing else about the gate changes.
//
// Consequences of that budget, both visible in the code below:
//   * no <regex>. The five patterns of the reference are hand-written matchers
//     (each one quotes the pattern it mirrors, so the pair can be diffed).
//     std::regex would cost construction time on the hot path, and MSVC's
//     backtracking matcher recurses -- a 10 000-character command is exactly
//     the input this gate must survive, not crash on.
//   * a hand-written JSON reader, for the same reason plus dependency-freedom:
//     the payload only ever needs three strings out of it.
//
// Install: build with cpp/build.py, which drops the binary next to hooks.json
// (see CMakeLists.txt). That location is not decoration -- `HERE` is the
// binary's own directory, and both `../bin/<tool>` and the wiring self-test
// resolve from it, exactly as they do for the script. Point the PreToolUse hook
// at the .exe instead of `python ...ask_user_gate.py` and nothing else changes.
//
// Standalone use (same verdict, exit 1 when denied):
//   ask_user_gate --check "git add -A && git commit -m x"
//   ask_user_gate --check "Get-Item a; Get-Item b" --shell powershell
//   ask_user_gate --check-file cmd.txt   // multi-line commands
//   ask_user_gate --self-test

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#else
#include <unistd.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

// ---------------------------------------------------------------------------
// Strings, paths and other small helpers
// ---------------------------------------------------------------------------

// fs::path's char constructor reads the NATIVE narrow encoding on Windows, not
// UTF-8, so a path with a non-ASCII component would round-trip wrong through
// std::string. Everything outside these two helpers is UTF-8 bytes.
fs::path pathFromUtf8(std::string_view text) {
	return fs::path(std::u8string(reinterpret_cast<const char8_t*>(text.data()), text.size()));
}

std::string pathToUtf8(const fs::path& path) {
	const std::u8string native = path.u8string();
	return std::string(reinterpret_cast<const char*>(native.data()), native.size());
}

bool startsWith(std::string_view text, std::string_view prefix) {
	return text.size() >= prefix.size() && text.compare(0, prefix.size(), prefix) == 0;
}

bool endsWith(std::string_view text, std::string_view suffix) {
	return text.size() >= suffix.size()
		&& text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string toLowerAscii(std::string_view text) {
	std::string out(text);
	for (char& c : out)
		if (c >= 'A' && c <= 'Z')
			c = static_cast<char>(c - 'A' + 'a');
	return out;
}

// Python's `\s`, and shlex's whitespace, over the bytes this scanner sees. A
// UTF-8 continuation byte is never one of these, so byte-wise scanning of a
// UTF-8 command is safe.
bool isSpaceChar(char c) {
	return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

bool isWordChar(char c) {
	const unsigned char u = static_cast<unsigned char>(c);
	return (u >= 'a' && u <= 'z') || (u >= 'A' && u <= 'Z') || (u >= '0' && u <= '9') || u == '_';
}

bool isDigitChar(char c) { return c >= '0' && c <= '9'; }

// `len(command)` in Python counts code points; std::string counts bytes. The
// two only differ above the analyser's limit and in reported offsets, but an
// off-by-a-few refusal reads as a port bug, so count what the reference counts.
size_t codePointCount(std::string_view text) {
	size_t count = 0;
	for (char c : text)
		if ((static_cast<unsigned char>(c) & 0xC0) != 0x80)
			++count;
	return count;
}

// fopen for a filesystem path. Narrow fopen would read a UTF-8 path in the ANSI
// codepage on Windows -- the same trap pathFromUtf8() exists for.
//
// stdio and not <fstream> throughout this file, for size: the iostream and
// locale machinery an ifstream drags into a statically linked binary is a third
// of the image, and /OPT:REF cannot drop it, because it IS referenced.
std::FILE* openFile(const fs::path& path, const char* mode) {
#ifdef _WIN32
	const std::wstring wideMode(mode, mode + std::strlen(mode));
	return _wfopen(path.c_str(), wideMode.c_str());
#else
	return std::fopen(path.c_str(), mode);
#endif
}

std::optional<std::string> readFile(const fs::path& path) {
	std::FILE* handle = openFile(path, "rb");
	if (!handle)
		return std::nullopt;
	std::string content;
	char buffer[8192];
	size_t got = 0;
	while ((got = std::fread(buffer, 1, sizeof(buffer), handle)) > 0)
		content.append(buffer, got);
	std::fclose(handle);
	return content;
}

// Absolute path of the running binary -- what the refusal names, and the anchor
// every other path in this file is measured from.
std::string executablePath() {
#ifdef _WIN32
	std::wstring buffer(32768, L'\0');
	const DWORD written = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
	buffer.resize(written);
	const int bytes = WideCharToMultiByte(CP_UTF8, 0, buffer.data(), static_cast<int>(buffer.size()),
		nullptr, 0, nullptr, nullptr);
	std::string utf8(static_cast<size_t>(bytes), '\0');
	WideCharToMultiByte(CP_UTF8, 0, buffer.data(), static_cast<int>(buffer.size()),
		utf8.data(), bytes, nullptr, nullptr);
	return utf8;
#else
	char buffer[4096];
	const ssize_t written = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
	if (written > 0) {
		buffer[written] = '\0';
		return std::string(buffer);
	}
	return pathToUtf8(fs::current_path() / "ask_user_gate");
#endif
}

std::string homeDirectory() {
#ifdef _WIN32
	if (const char* profile = std::getenv("USERPROFILE"))
		return profile;
	const char* drive = std::getenv("HOMEDRIVE");
	const char* rest = std::getenv("HOMEPATH");
	if (drive && rest)
		return std::string(drive) + rest;
	return {};
#else
	if (const char* home = std::getenv("HOME"))
		return home;
	return {};
#endif
}

bool isWindowsHost() {
#ifdef _WIN32
	return true;
#else
	return false;
#endif
}

const std::string kSelf = executablePath();
const std::string kHere = pathToUtf8(pathFromUtf8(kSelf).parent_path());

// ---------------------------------------------------------------------------
// Constants mirrored from the reference
// ---------------------------------------------------------------------------

// The scripts the refusals below send the caller to. They ship in ../bin.
constexpr std::string_view kShippedTools[] = {"replace_in_file.py", "try_patch.py"};

// What a `tools/<name>.py` in a checkout must contain to be recognised as a
// stand-in for one of those rather than an unrelated script of the same name.
constexpr std::string_view kToolMarker = "ask-user-gate";

// The marker that turns the gate off for one command, matched case-insensitively.
constexpr std::string_view kMarker = "allowaskuser";

// Hard limit of the command analyser inside Claude Code (MAX_COMMAND_LENGTH).
constexpr size_t kMaxCommandLength = 10000;

constexpr std::string_view kBashTools[] = {"Bash"};
constexpr std::string_view kPowerShellTools[] = {"PowerShell"};
// Monitor carries a `command` of its own and runs it in the same local shell,
// under a tool name allow-rules written `Bash(...)` do not cover.
constexpr std::string_view kMonitorTools[] = {"Monitor"};

template <size_t N>
bool contains(const std::string_view (&list)[N], std::string_view value) {
	for (std::string_view item : list)
		if (item == value)
			return true;
	return false;
}

// The shell's working directory, from the hook payload. Empty in standalone
// use, where the process's own directory is the same thing. See callerDir().
std::string gCallerCwd;

constexpr std::string_view kSleepFix = R"GATE(wait with a tool, not with the clock. Three replacements, in order:
       (1) run the long command in the FOREGROUND with the tool's `timeout` parameter (up to 600000 ms) -- the call blocks for you, so there is nothing left to sleep for;
       (2) wait for a CONDITION with the Monitor tool (it is deferred: run ToolSearch("select:Monitor") first, then give it an until-loop);
       (3) start the work with the tool's `run_in_background` parameter and let the completion notification reach you.
       A SUBAGENT is not woken by its own background job -- only the main loop is -- so for a subagent (3) means no wakeup at all: run it in the foreground instead. A wait longer than the 10-minute foreground cap is the orchestrator's work, not the subagent's: report and hand it back.
       If you truly must idle, one long call costs one prompt and a loop of short ones costs a prompt per iteration.)GATE";

constexpr std::string_view kMonitorSleepFix = R"GATE(a one-shot wait (`break` when the thing is done) is not what Monitor is for -- its own description sends that case to a foreground call or to Bash `run_in_background`.
       (1) run the work itself in the FOREGROUND with the tool's `timeout` parameter (up to 600000 ms) instead of starting it detached and watching for its death;
       (2) if you are a SUBAGENT, this is the only option: background events do not re-invoke you, so a Monitor armed by a subagent notifies nobody and only burns its timeout. A wait longer than the foreground cap is the orchestrator's work -- report and hand it back.
       Note also that allow-rules are written `Bash(...)` and do not cover Monitor, so its command asks the human even when the same text would have been allowed as a Bash call.)GATE";

constexpr std::string_view kChainFix = R"GATE(this chain also moves the working directory, so where the rest of it runs stops being a static fact and no relative path in it can be resolved. Name the directory with the tool's own flag (--cwd / --prefix / -C) or an absolute path instead of `cd X && ...`, and keep one command per call.)GATE";

constexpr std::string_view kChainFixWindowsNote = R"GATE( On Windows this one is unappealable: the write target is relative, and under Git Bash the working directory a chain ends in cannot be determined statically, so the analyser cannot check that target and refuses to delegate the decision at all.)GATE";

constexpr std::string_view kBraceFix = R"GATE(keep the payload out of the command line: write it to a file with the Write tool and let the command be a call to that file, or quote the whole argument so the braces sit inside a string.)GATE";

constexpr std::string_view kBackgroundFix = R"GATE(use the tool's run_in_background parameter instead of detaching with `&`.)GATE";

constexpr std::string_view kLengthFix = R"GATE(above that limit the command cannot be parsed at all and the decision is forced to 'ask the human' whatever the allow-rules say. Move the payload into a file written with the Write tool and keep the command a short call to it.)GATE";

// ---------------------------------------------------------------------------
// Where the refusals point
// ---------------------------------------------------------------------------

// The directory a relative path in a refusal will be resolved against: the
// shell's, from the payload, not the project root.
std::string callerDir() {
	if (!gCallerCwd.empty())
		return gCallerCwd;
	return pathToUtf8(fs::current_path());
}

// Is that OUR script, or something else that happens to share its name?
bool isOurCopy(const fs::path& path) {
	std::FILE* handle = openFile(path, "rb");
	if (!handle)
		return false;
	std::string head(4096, '\0');
	const size_t got = std::fread(head.data(), 1, head.size(), handle);
	std::fclose(handle);
	head.resize(got);
	return head.find(kToolMarker) != std::string::npos;
}

// Where one of kShippedTools sits on THIS machine, absolute. The layout
// `hooks/<this binary>` beside `bin/<tool>` is a fact about the plugin, so the
// `..` hop is too; checkPaths() asks the filesystem whether it still holds.
std::string shippedPath(std::string_view name) {
	return pathToUtf8((pathFromUtf8(kHere) / ".." / "bin" / pathFromUtf8(name)).lexically_normal());
}

// How to spell one of kShippedTools so that THIS machine can run it. A checkout
// that keeps its own stand-ins under `tools/` gets the short, already-familiar
// path -- but only when that path resolves from where the command will actually
// run. Anywhere else gets the absolute path to the copy that shipped here.
std::string toolPath(std::string_view name, std::optional<std::string> base = std::nullopt) {
	const std::string resolved = base ? *base : callerDir();
	if (isOurCopy(pathFromUtf8(resolved) / "tools" / pathFromUtf8(name)))
		return std::string("tools/") + std::string(name);
	return shippedPath(name);
}

// ---------------------------------------------------------------------------
// The five patterns of the reference, hand-written
//
// Each function names the Python regex it mirrors and repeats its source, so
// the pair can be diffed by eye; parity_check.py diffs them by running both.
// None of them backtracks: every `\s*` here is followed by something that
// cannot be whitespace, so max-munch is the only viable split.
// ---------------------------------------------------------------------------

size_t skipSpaces(std::string_view text, size_t index) {
	while (index < text.size() && isSpaceChar(text[index]))
		++index;
	return index;
}

// The `(?:^|[;&|(){}\n])` alternative shared by CD_COMMAND and the sleep
// patterns: does a top-level separator end right before `index`?
bool afterSeparator(std::string_view text, size_t index) {
	if (index == 0)
		return true;
	const char previous = text[index - 1];
	return previous == ';' || previous == '&' || previous == '|' || previous == '('
		|| previous == ')' || previous == '{' || previous == '}' || previous == '\n';
}

// The extra `|\b(?:do|then|else)\s` alternative of the sleep patterns. It
// CONSUMES the whitespace, so the anchor ends just past it -- which is how
// `; do sleep 30` is caught, the shape that smuggles a foreground `sleep` past
// the Bash tool's own block on it.
bool afterLoopKeyword(std::string_view text, size_t index) {
	if (index == 0 || !isSpaceChar(text[index - 1]))
		return false;
	const size_t wordEnd = index - 1;
	for (std::string_view keyword : {std::string_view("do"), std::string_view("then"),
			std::string_view("else")}) {
		if (wordEnd < keyword.size())
			continue;
		const size_t wordStart = wordEnd - keyword.size();
		if (text.compare(wordStart, keyword.size(), keyword) != 0)
			continue;
		if (wordStart > 0 && isWordChar(text[wordStart - 1]))
			continue;  // the `\b` before the keyword
		return true;
	}
	return false;
}

// CD_COMMAND:
//   (?:^|[;&|(){}\n])\s*(?:cd|pushd|popd|chdir|[Ss]et-[Ll]ocation|sl)(?:\s|$)
bool matchesCdCommand(std::string_view text) {
	static constexpr std::string_view words[] = {"cd", "pushd", "popd", "chdir",
		"Set-Location", "Set-location", "set-Location", "set-location", "sl"};
	for (size_t index = 0; index <= text.size(); ++index) {
		if (!afterSeparator(text, index))
			continue;
		const size_t start = skipSpaces(text, index);
		for (std::string_view word : words) {
			if (text.size() - start < word.size())
				continue;
			if (text.compare(start, word.size(), word) != 0)
				continue;
			const size_t after = start + word.size();
			if (after == text.size() || isSpaceChar(text[after]))
				return true;
		}
	}
	return false;
}

// SLEEP_COMMAND:
//   (?:^|[;&|(){}\n]|\b(?:do|then|else)\s)\s*sleep\b\s*[\d$]
bool matchesSleepCommand(std::string_view text) {
	for (size_t index = 0; index <= text.size(); ++index) {
		if (!afterSeparator(text, index) && !afterLoopKeyword(text, index))
			continue;
		const size_t start = skipSpaces(text, index);
		if (text.size() - start < 5 || text.compare(start, 5, "sleep") != 0)
			continue;
		const size_t after = start + 5;
		if (after < text.size() && isWordChar(text[after]))
			continue;  // the `\b` after the word
		const size_t argument = skipSpaces(text, after);
		if (argument < text.size() && (isDigitChar(text[argument]) || text[argument] == '$'))
			return true;
	}
	return false;
}

// START_SLEEP_COMMAND (case-insensitive):
//   (?:^|[;&|(){}\n]|\b(?:do|then|else)\s)\s*start-sleep\b
bool matchesStartSleepCommand(std::string_view text) {
	static constexpr std::string_view word = "start-sleep";
	for (size_t index = 0; index <= text.size(); ++index) {
		if (!afterSeparator(text, index) && !afterLoopKeyword(text, index))
			continue;
		const size_t start = skipSpaces(text, index);
		if (text.size() - start < word.size())
			continue;
		if (toLowerAscii(text.substr(start, word.size())) != word)
			continue;
		const size_t after = start + word.size();
		if (after < text.size() && isWordChar(text[after]))
			continue;
		return true;
	}
	return false;
}

// RELATIVE_REDIRECT:
//   >>?\s*(?!/|[A-Za-z]:[\\/]|&)([^\s|;&<>()]+)
// Only used to explain the chain finding better on Windows.
bool matchesRelativeRedirect(std::string_view text) {
	auto excluded = [](char c) {
		return isSpaceChar(c) || c == '|' || c == ';' || c == '&' || c == '<' || c == '>'
			|| c == '(' || c == ')';
	};
	for (size_t index = 0; index < text.size(); ++index) {
		if (text[index] != '>')
			continue;
		// `>>?` is greedy, so the two-character form is tried first.
		for (size_t arrow : {size_t(2), size_t(1)}) {
			if (arrow == 2 && (index + 1 >= text.size() || text[index + 1] != '>'))
				continue;
			const size_t target = skipSpaces(text, index + arrow);
			if (target >= text.size())
				continue;
			if (text[target] == '/' || text[target] == '&')
				continue;  // absolute, or `2>&1`
			const bool driveLetter = target + 2 < text.size()
				&& isWordChar(text[target]) && !isDigitChar(text[target]) && text[target] != '_'
				&& text[target + 1] == ':' && (text[target + 2] == '\\' || text[target + 2] == '/');
			if (driveLetter)
				continue;
			if (!excluded(text[target]))
				return true;
		}
	}
	return false;
}

// ---------------------------------------------------------------------------
// The scanner
// ---------------------------------------------------------------------------

struct Finding {
	std::string reason;
	std::string fix;
	// Chain findings survive only when the command also moves the working
	// directory; everything else is refused on its own.
	bool chain = false;
};

// Walk the command outside quotes, reporting separators and heredocs. Quote
// tracking is what makes this usable: `python -c 'a; b'` keeps its semicolon
// inside a string and is not a chain, while `a; b` is. The escape character
// differs per shell -- backslash in sh, backtick in PowerShell, where a
// trailing backslash in a Windows path would otherwise eat the closing quote
// and desynchronise the whole scan.
std::vector<Finding> scanShellSyntax(const std::string& command, std::string_view shell,
		bool windows, std::string_view sleepFix) {
	const char escape = (shell == "bash") ? '\\' : '`';
	std::vector<Finding> findings;
	std::vector<std::string> seen;

	auto report = [&](std::string_view token, size_t index, std::string reason,
			std::string fix, bool chain = false) {
		// One line per kind of violation, not per occurrence.
		std::string key(token);
		if (std::find(seen.begin(), seen.end(), key) != seen.end())
			return;
		seen.push_back(std::move(key));
		reason += " (at offset " + std::to_string(codePointCount(std::string_view(command).substr(0, index))) + ")";
		findings.push_back(Finding{std::move(reason), std::move(fix), chain});
	};

	std::string chainFix(kChainFix);
	if (windows && matchesRelativeRedirect(command))
		chainFix += kChainFixWindowsNote;
	const std::string heredocFix = "write files with the Write tool, edit them with Edit, and do "
		"scripted replacements with " + toolPath("replace_in_file.py") + ". If the content must "
		"be produced by a program, let the script write the file and keep the command a call to "
		"that script.";

	size_t i = 0;
	const size_t n = command.size();
	char quote = '\0';
	int braces = 0;              // depth of `{ ... }` groups seen OUTSIDE quotes
	std::string plain;           // the command with quoted runs blanked out
	plain.reserve(n);
	while (i < n) {
		const char c = command[i];
		if (quote != '\0') {
			if (c == escape && quote == '"' && i + 1 < n) {
				i += 2;
				continue;
			}
			if (c == quote)
				quote = '\0';
			i += 1;
			continue;
		}
		// From here on we are at top level, so `plain` gets this character -- a
		// quoted run collapses to one space, which keeps words apart without
		// letting a string body look like a command word.
		plain.push_back((c == '\'' || c == '"') ? ' ' : c);
		if (c == escape && i + 1 < n) {
			i += 2;  // escaped char, including a line continuation
			continue;
		}
		if (c == '\'' || c == '"') {
			if (braces > 0) {
				// The analyser reads a quote inside unquoted braces as expansion
				// obfuscation and asks a human. `awk '{print "x"}'` is fine (the
				// braces are inside the quotes, not the other way round); a
				// heredoc body or a bare ${VAR:-"x"} is not.
				report("{", i, "quote character inside unquoted `{ ... }` -- read as expansion "
					"obfuscation", std::string(kBraceFix));
			}
			quote = c;
			i += 1;
			continue;
		}
		if (c == '{') {
			braces += 1;
			i += 1;
			continue;
		}
		if (c == '}') {
			braces = std::max(0, braces - 1);
			i += 1;
			continue;
		}
		if (c == '#' && (i == 0 || command[i - 1] == ' ' || command[i - 1] == '\t'
				|| command[i - 1] == '\n')) {
			const size_t newline = command.find('\n', i);
			if (newline == std::string::npos)
				break;
			i = newline;  // the newline itself is still a top-level separator
			continue;
		}

		const std::string_view two = std::string_view(command).substr(i, 2);
		if (two == "&&" || two == "||") {
			report(two, i, "shell chain operator `" + std::string(two) + "` in a chain that "
				"changes directory", chainFix, true);
			i += 2;
			continue;
		}
		if (two == "<<") {
			if (std::string_view(command).substr(i, 3) == "<<<") {
				i += 3;  // here-string: one line, not a file body
				continue;
			}
			report("<<", i, "heredoc (`<<`) -- a file written by the command body", heredocFix);
			i += 2;
			continue;
		}
		if ((two == "@'" || two == "@\"") && shell == "powershell") {
			report("@'", i, "PowerShell here-string -- a file written by the command body",
				heredocFix);
			i += 2;
			continue;
		}
		if (c == ';') {
			report(";", i, "command separator `;` in a chain that changes directory", chainFix, true);
			i += 1;
			continue;
		}
		if (c == '\n') {
			report("\n", i, "newline outside quotes -- a second command on its own line, in a "
				"chain that changes directory", chainFix, true);
			i += 1;
			continue;
		}
		if (c == '&') {
			// `2>&1`, `&>file`, `|&` are redirections, not backgrounding.
			const char previous = i ? command[i - 1] : '\0';
			const bool redirect = previous == '>' || previous == '<' || previous == '|'
				|| (i + 1 < n && command[i + 1] == '>');
			if (!redirect)
				report("&", i, "backgrounding `&`", std::string(kBackgroundFix));
			i += 1;
			continue;
		}
		i += 1;
	}

	// Offsets are deliberately absent here: `plain` drops quoted runs rather
	// than blanking them, so its indices are not the command's.
	if (matchesSleepCommand(plain) || matchesStartSleepCommand(plain))
		findings.push_back(Finding{"waiting by the clock (`sleep`)", std::string(sleepFix), false});
	if (!matchesCdCommand(plain)) {
		// A chain of allow-listed commands appears to be matched fine; it is the
		// directory change that defeats the analyser. See CD_COMMAND.
		std::erase_if(findings, [](const Finding& finding) { return finding.chain; });
	}
	return findings;
}

// shlex.split(command, posix=True). Raises on an unterminated quote or a
// dangling escape, which the caller answers with a plain whitespace split --
// the same fallback the reference takes.
bool shlexSplit(const std::string& command, std::vector<std::string>& tokens) {
	tokens.clear();
	std::string token;
	bool started = false;  // a quoted empty string is still a token
	size_t i = 0;
	const size_t n = command.size();
	while (i < n) {
		const char c = command[i];
		if (isSpaceChar(c)) {
			if (started) {
				tokens.push_back(token);
				token.clear();
				started = false;
			}
			++i;
			continue;
		}
		started = true;
		if (c == '\\') {
			if (i + 1 >= n)
				return false;  // ValueError: No escaped character
			token.push_back(command[i + 1]);
			i += 2;
			continue;
		}
		if (c == '\'') {
			const size_t close = command.find('\'', i + 1);
			if (close == std::string::npos)
				return false;  // ValueError: No closing quotation
			token.append(command, i + 1, close - i - 1);
			i = close + 1;
			continue;
		}
		if (c == '"') {
			++i;
			bool closed = false;
			while (i < n) {
				if (command[i] == '"') {
					closed = true;
					++i;
					break;
				}
				if (command[i] == '\\') {
					if (i + 1 >= n)
						return false;
					// Inside double quotes a backslash escapes only `"` and `\`;
					// anything else keeps the backslash too.
					const char next = command[i + 1];
					if (next != '"' && next != '\\' && next != '\'')
						token.push_back('\\');
					token.push_back(next);
					i += 2;
					continue;
				}
				token.push_back(command[i]);
				++i;
			}
			if (!closed)
				return false;
			continue;
		}
		token.push_back(c);
		++i;
	}
	if (started)
		tokens.push_back(token);
	return true;
}

void whitespaceSplit(const std::string& command, std::vector<std::string>& tokens) {
	tokens.clear();
	size_t i = 0;
	while (i < command.size()) {
		while (i < command.size() && isSpaceChar(command[i]))
			++i;
		const size_t start = i;
		while (i < command.size() && !isSpaceChar(command[i]))
			++i;
		if (i > start)
			tokens.push_back(command.substr(start, i - start));
	}
}

// `sed -i` -- banned by CLAUDE.md and not on any allow-list here.
std::vector<Finding> scanSedInPlace(const std::string& command) {
	std::vector<std::string> tokens;
	if (!shlexSplit(command, tokens))
		whitespaceSplit(command, tokens);
	for (size_t index = 0; index < tokens.size(); ++index) {
		const std::string& token = tokens[index];
		if (token != "sed" && !endsWith(token, "/sed"))
			continue;
		for (size_t rest = index + 1; rest < tokens.size(); ++rest) {
			const std::string& arg = tokens[rest];
			if (arg == "|" || arg == ";" || arg == "&&" || arg == "||")
				break;
			if (arg != "--in-place" && !startsWith(arg, "--in-place=")) {
				// A short-flag cluster carrying `i`, with `-i.bak`'s suffix cut off.
				std::string head = arg.substr(0, arg.find('.'));
				const size_t dashes = head.find_first_not_of('-');
				head = (dashes == std::string::npos) ? std::string() : head.substr(dashes);
				const bool shortFlagWithI = arg.size() > 1 && arg[0] == '-' && arg[1] != '-'
					&& head.find('i') != std::string::npos;
				if (!shortFlagWithI)
					continue;
			}
			return {Finding{"`sed -i`",
				"use " + toolPath("replace_in_file.py") + " (it refuses to write when the match "
				"count is not the one you named, so a wrong pattern cannot pass as a successful "
				"edit), or " + toolPath("try_patch.py") + " when the edit is temporary and must "
				"be undone after a test run.", false}};
		}
	}
	return {};
}

// All reasons this command would stop for a human, or an empty list.
std::vector<Finding> scan(const std::string& command, std::string_view shell,
		std::optional<bool> windows = std::nullopt, std::string_view tool = "Bash") {
	if (toLowerAscii(command).find(kMarker) != std::string::npos)
		return {};
	const bool onWindows = windows ? *windows : isWindowsHost();
	const std::string_view sleepFix = contains(kMonitorTools, tool) ? kMonitorSleepFix : kSleepFix;
	std::vector<Finding> findings;
	const size_t length = codePointCount(command);
	if (length > kMaxCommandLength) {
		findings.push_back(Finding{
			"the command is " + std::to_string(length) + " characters, over the analyser's "
			+ std::to_string(kMaxCommandLength) + "-character limit",
			std::string(kLengthFix), false});
	}
	const std::vector<Finding> syntax = scanShellSyntax(command, shell, onWindows, sleepFix);
	findings.insert(findings.end(), syntax.begin(), syntax.end());
	if (shell == "bash") {
		const std::vector<Finding> sed = scanSedInPlace(command);
		findings.insert(findings.end(), sed.begin(), sed.end());
	}
	return findings;
}

// The refusal the agent reads, in place of the dialog the user would.
std::string render(const std::vector<Finding>& findings) {
	// Resolved, not hardcoded: the same gate is a plugin on one machine and a
	// loose hook on another, and a reader who has never seen this plugin needs
	// to know where the thing refusing their command lives.
	std::string out = "Blocked by " + kSelf + ": this command would stop the session on a "
		"permission prompt, so it was not run.\n\n";
	std::vector<std::string> said;  // several findings usually share one remedy
	for (const Finding& finding : findings) {
		out += "  - " + finding.reason + "\n";
		if (std::find(said.begin(), said.end(), finding.fix) == said.end()) {
			said.push_back(finding.fix);
			out += "    -> " + finding.fix + "\n";
		}
	}
	out += "\nIf you want this exact command anyway and accept that the user will be asked, add "
		"the marker `allowAskUser` to it (e.g. append ` # allowAskUser`) and it will be passed "
		"through unchanged.";
	return out;
}

std::optional<std::string> shellForTool(std::string_view toolName) {
	if (contains(kBashTools, toolName) || contains(kMonitorTools, toolName))
		return std::string("bash");
	if (contains(kPowerShellTools, toolName))
		return std::string("powershell");
	return std::nullopt;
}

// ---------------------------------------------------------------------------
// JSON: three strings out of the payload, a handful out of the hook configs,
// and one object back. Hand-written to keep the process free of dependencies
// and of anything that has to be constructed before the first useful byte.
// ---------------------------------------------------------------------------

struct Json {
	enum class Type { Null, Bool, Number, String, Array, Object };
	Type type = Type::Null;
	bool boolean = false;
	double number = 0;
	std::string text;
	std::vector<Json> items;
	std::vector<std::pair<std::string, Json>> members;

	const Json* member(std::string_view key) const {
		if (type != Type::Object)
			return nullptr;
		for (const auto& entry : members)
			if (entry.first == key)
				return &entry.second;
		return nullptr;
	}
	// The value at `key` if it is a string, else an empty string. Mirrors the
	// reference's `.get(key, "")` -- a payload of the wrong shape is not an
	// error, it is a call this gate has nothing to say about.
	std::string memberString(std::string_view key) const {
		const Json* found = member(key);
		return (found && found->type == Type::String) ? found->text : std::string();
	}
};

class JsonParser {
public:
	explicit JsonParser(std::string_view source) : source_(source) {}

	// Returns false on malformed input rather than throwing: every caller
	// answers a parse failure with "leave this call alone".
	bool parse(Json& out) {
		skipSpace();
		if (!parseValue(out))
			return false;
		skipSpace();
		return index_ >= source_.size();
	}

private:
	std::string_view source_;
	size_t index_ = 0;

	void skipSpace() {
		while (index_ < source_.size() && isSpaceChar(source_[index_]))
			++index_;
	}

	bool literal(std::string_view word) {
		if (source_.compare(index_, word.size(), word) != 0)
			return false;
		index_ += word.size();
		return true;
	}

	static void appendUtf8(std::string& out, unsigned int code) {
		if (code < 0x80) {
			out.push_back(static_cast<char>(code));
		} else if (code < 0x800) {
			out.push_back(static_cast<char>(0xC0 | (code >> 6)));
			out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
		} else if (code < 0x10000) {
			out.push_back(static_cast<char>(0xE0 | (code >> 12)));
			out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
			out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
		} else {
			out.push_back(static_cast<char>(0xF0 | (code >> 18)));
			out.push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3F)));
			out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
			out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
		}
	}

	bool parseHex4(unsigned int& value) {
		if (index_ + 4 > source_.size())
			return false;
		value = 0;
		for (int digit = 0; digit < 4; ++digit) {
			const char c = source_[index_ + static_cast<size_t>(digit)];
			value <<= 4;
			if (c >= '0' && c <= '9')
				value |= static_cast<unsigned int>(c - '0');
			else if (c >= 'a' && c <= 'f')
				value |= static_cast<unsigned int>(c - 'a' + 10);
			else if (c >= 'A' && c <= 'F')
				value |= static_cast<unsigned int>(c - 'A' + 10);
			else
				return false;
		}
		index_ += 4;
		return true;
	}

	bool parseString(std::string& out) {
		if (index_ >= source_.size() || source_[index_] != '"')
			return false;
		++index_;
		while (index_ < source_.size()) {
			const char c = source_[index_];
			if (c == '"') {
				++index_;
				return true;
			}
			if (c != '\\') {
				out.push_back(c);
				++index_;
				continue;
			}
			++index_;
			if (index_ >= source_.size())
				return false;
			const char escape = source_[index_++];
			switch (escape) {
			case '"': out.push_back('"'); break;
			case '\\': out.push_back('\\'); break;
			case '/': out.push_back('/'); break;
			case 'b': out.push_back('\b'); break;
			case 'f': out.push_back('\f'); break;
			case 'n': out.push_back('\n'); break;
			case 'r': out.push_back('\r'); break;
			case 't': out.push_back('\t'); break;
			case 'u': {
				unsigned int code = 0;
				if (!parseHex4(code))
					return false;
				if (code >= 0xD800 && code <= 0xDBFF && index_ + 1 < source_.size()
						&& source_[index_] == '\\' && source_[index_ + 1] == 'u') {
					const size_t saved = index_;
					index_ += 2;
					unsigned int low = 0;
					if (parseHex4(low) && low >= 0xDC00 && low <= 0xDFFF)
						code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
					else
						index_ = saved;
				}
				appendUtf8(out, code);
				break;
			}
			default:
				return false;
			}
		}
		return false;
	}

	bool parseValue(Json& out) {
		if (index_ >= source_.size())
			return false;
		const char c = source_[index_];
		if (c == '"') {
			out.type = Json::Type::String;
			return parseString(out.text);
		}
		if (c == '{') {
			out.type = Json::Type::Object;
			++index_;
			skipSpace();
			if (index_ < source_.size() && source_[index_] == '}') {
				++index_;
				return true;
			}
			while (true) {
				skipSpace();
				std::string key;
				if (!parseString(key))
					return false;
				skipSpace();
				if (index_ >= source_.size() || source_[index_] != ':')
					return false;
				++index_;
				skipSpace();
				Json value;
				if (!parseValue(value))
					return false;
				out.members.emplace_back(std::move(key), std::move(value));
				skipSpace();
				if (index_ < source_.size() && source_[index_] == ',') {
					++index_;
					continue;
				}
				if (index_ < source_.size() && source_[index_] == '}') {
					++index_;
					return true;
				}
				return false;
			}
		}
		if (c == '[') {
			out.type = Json::Type::Array;
			++index_;
			skipSpace();
			if (index_ < source_.size() && source_[index_] == ']') {
				++index_;
				return true;
			}
			while (true) {
				skipSpace();
				Json value;
				if (!parseValue(value))
					return false;
				out.items.push_back(std::move(value));
				skipSpace();
				if (index_ < source_.size() && source_[index_] == ',') {
					++index_;
					continue;
				}
				if (index_ < source_.size() && source_[index_] == ']') {
					++index_;
					return true;
				}
				return false;
			}
		}
		if (c == 't') {
			out.type = Json::Type::Bool;
			out.boolean = true;
			return literal("true");
		}
		if (c == 'f') {
			out.type = Json::Type::Bool;
			out.boolean = false;
			return literal("false");
		}
		if (c == 'n') {
			out.type = Json::Type::Null;
			return literal("null");
		}
		out.type = Json::Type::Number;
		const char* begin = source_.data() + index_;
		char* end = nullptr;
		out.number = std::strtod(begin, &end);
		if (end == begin)
			return false;
		index_ += static_cast<size_t>(end - begin);
		return true;
	}
};

std::string jsonEscape(std::string_view text) {
	std::string out;
	out.reserve(text.size() + 16);
	for (char c : text) {
		switch (c) {
		case '"': out += "\\\""; break;
		case '\\': out += "\\\\"; break;
		case '\n': out += "\\n"; break;
		case '\r': out += "\\r"; break;
		case '\t': out += "\\t"; break;
		case '\b': out += "\\b"; break;
		case '\f': out += "\\f"; break;
		default:
			if (static_cast<unsigned char>(c) < 0x20) {
				char buffer[7];
				std::snprintf(buffer, sizeof(buffer), "\\u%04x", static_cast<unsigned char>(c));
				out += buffer;
			} else {
				out.push_back(c);
			}
		}
	}
	return out;
}

// ---------------------------------------------------------------------------
// Self-test
// ---------------------------------------------------------------------------

struct SelfTestCase {
	std::string command;
	std::string shell;
	bool denied;
	std::string tool = "Bash";
};

std::vector<SelfTestCase> selfTestCases() {
	return {
		{"git status", "bash", false},
		{"npm run build --prefix webgame", "bash", false},
		{"git log --oneline -20 | head -5", "bash", false},
		{"python -c 'print(1); print(2)'", "bash", false},
		{"grep -rn 'a && b' src", "bash", false},
		{"node -e \"console.log(1)\" 2>&1", "bash", false},
		// A plain chain is let through; a chain that moves the cwd is not.
		{"git add -A && git commit -m x", "bash", false},
		{"ls; pwd", "bash", false},
		{"rg --version; rg --files | head -2", "bash", false},
		{"cd webgame && npx vitest run", "bash", true},
		{"cd webgame; npx vitest run", "bash", true},
		{"pushd webgame && npm run build", "bash", true},
		{"echo 'cd webgame && x'", "bash", false},
		{"git log --format=%cd; git status", "bash", false},
		{"Set-Location webgame; npx vitest run", "powershell", true},
		{"npm run dev &", "bash", true},
		{"cat > f.txt <<'EOF'\nbody\nEOF", "bash", true},
		{"ls a <<< 'x'", "bash", false},
		{"sed -i 's/a/b/' file.ts", "bash", true},
		{"sed -i.bak 's/a/b/' file.ts", "bash", true},
		{"sed -n '1,5p' file.ts", "bash", false},
		{"echo 'sed -i is banned'", "bash", false},
		{std::string(kMaxCommandLength + 1, 'x'), "bash", true},
		// Waiting by the clock, including the loop that smuggles a foreground
		// `sleep` past the Bash tool's own block on it.
		{"sleep 30", "bash", true},
		{"i=0; while [ $i -lt 55 ]; do sleep 30; i=$((i+1)); done", "bash", true},
		{"while true; do sleep $DELAY; done", "bash", true},
		{"npm run dev --prefix webgame; sleep 2", "bash", true},
		{"Start-Sleep -Seconds 30", "powershell", true},
		{"sleep 5", "powershell", true},
		// ... and the words that only look like it.
		{"ls sleep.txt", "bash", false},
		{"npm run probe --prefix webgame -- --sleep 5", "bash", false},
		{"grep -n 'sleep 30' scripts/probe.sh", "bash", false},
		{"python -c \"import time; time.sleep(1)\"", "bash", false},
		{"awk '{print \"x\"}' file", "bash", false},
		{"find . -name '*.tmp' -exec rm {} +", "bash", false},
		{"echo ${VAR:-\"x\"}", "bash", true},
		{"git add -A && git commit -m x  # allowAskUser", "bash", false},
		{"Get-ChildItem C:\\game\\1339", "powershell", false},
		{"Get-Item a; Get-Item b", "powershell", false},
		{"git commit -F @'\nmsg\n'@", "powershell", true},
		// Monitor carries a shell command under a tool name of its own. The
		// first case is the one that walked past this gate into a dialog.
		{"while true; do if ! tasklist //FI \"IMAGENAME eq node.exe\" | grep -qi node.exe; then "
			"echo done; break; fi; sleep 10; done", "bash", true, "Monitor"},
		{"tail -f webgame/.devlogs/latest.log | grep --line-buffered ERROR", "bash", false, "Monitor"},
	};
}

std::vector<std::string> handledTools() {
	return {"Bash", "Monitor", "PowerShell"};  // sorted, as in the reference
}

// `(?:^|\|)<tool>(?:\||$)` over a hook matcher, written out.
bool matcherNamesTool(std::string_view matcher, std::string_view tool) {
	size_t start = 0;
	while (start <= matcher.size()) {
		const size_t bar = matcher.find('|', start);
		const size_t end = (bar == std::string_view::npos) ? matcher.size() : bar;
		if (matcher.substr(start, end - start) == tool)
			return true;
		if (bar == std::string_view::npos)
			break;
		start = bar + 1;
	}
	return false;
}

// Every tool this binary handles must also be in the hook's matcher.
//
// The failure this pins is the one that happened: the scanner refused the
// command in every standalone check, and the session still stopped on a dialog,
// because the tool that carried it (Monitor) was not in the matcher of the
// PreToolUse entry -- so the hook never ran. EVERY file that could be running it
// is checked, not the first one found: checking only one recreates the hole on a
// machine where the LIVE wiring is the other file.
std::pair<std::vector<std::string>, int> checkWiring() {
	std::vector<std::string> problems;
	const std::vector<std::string> handled = handledTools();
	int checks = static_cast<int>(handled.size()) + 1;
	for (const std::string& tool : handled)
		if (!shellForTool(tool))
			problems.push_back("shellForTool(" + tool + ") is empty");
	if (shellForTool("Read"))
		problems.push_back("shellForTool routes a tool that carries no command");

	// A wiring written for the SCRIPT names ask_user_gate.py and one written for
	// this binary names ask_user_gate.exe, and either is a live gate. Matching on
	// the stem is what lets the same self-test answer the question -- "is the
	// matcher complete?" -- whichever copy is wired here.
	const std::string stem = pathToUtf8(pathFromUtf8(kSelf).stem());

	std::vector<std::string> candidates;
	auto addCandidate = [&](const fs::path& path) {
		std::string text = pathToUtf8(path);
		if (text.empty())
			return;
		if (std::find(candidates.begin(), candidates.end(), text) == candidates.end())
			candidates.push_back(std::move(text));
	};
	addCandidate(pathFromUtf8(kHere) / "hooks.json");
	addCandidate(pathFromUtf8(kHere) / "settings.json");
	addCandidate(pathFromUtf8(callerDir()) / ".claude" / "settings.json");
	const std::string home = homeDirectory();
	if (!home.empty())
		addCandidate(pathFromUtf8(home) / ".claude" / "settings.json");

	int wired = 0;
	for (const std::string& config : candidates) {
		const fs::path path = pathFromUtf8(config);
		std::error_code error;
		if (!fs::is_regular_file(path, error))
			continue;
		checks += 1;
		const std::optional<std::string> content = readFile(path);
		Json root;
		if (!content || !JsonParser(*content).parse(root)) {
			problems.push_back("cannot read " + config);
			continue;
		}
		const Json* hooks = root.member("hooks");
		const Json* preToolUse = hooks ? hooks->member("PreToolUse") : nullptr;
		if (!preToolUse || preToolUse->type != Json::Type::Array)
			continue;
		for (const Json& entry : preToolUse->items) {
			const Json* entryHooks = entry.member("hooks");
			bool ours = false;
			if (entryHooks && entryHooks->type == Json::Type::Array)
				for (const Json& hook : entryHooks->items)
					if (hook.memberString("command").find(stem) != std::string::npos)
						ours = true;
			if (!ours)
				continue;
			wired += 1;
			const std::string matcher = entry.memberString("matcher");
			checks += static_cast<int>(handled.size());
			for (const std::string& tool : handled)
				if (!matcherNamesTool(matcher, tool))
					problems.push_back(tool + " is handled here but missing from the matcher '"
						+ matcher + "' in " + config);
		}
	}
	if (wired == 0) {
		std::string names;
		for (size_t i = 0; i < candidates.size(); ++i)
			names += (i ? ", " : "") + candidates[i];
		problems.push_back("nothing runs this gate: no PreToolUse entry names " + stem
			+ " in any of " + names);
	}
	return {problems, checks};
}

// Both spellings toolPath() can produce must land on a real file.
//
// Without this the resolver is untestable in the only way that matters: its
// output is prose inside a refusal, which nothing compiles and nobody diffs, so
// a rename in ../bin or a wrong `..` hop would surface as an agent searching for
// a file that is not there. The last case is the one that earns the rest --
// `base` unset, the only spelling the hook ever uses.
std::pair<std::vector<std::string>, int> checkPaths() {
	std::vector<std::string> problems;
	int checks = 0;
	std::error_code error;
	fs::path root;
	for (int attempt = 0; attempt < 8; ++attempt) {
		const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
		root = fs::temp_directory_path(error) / ("ask_user_gate_selftest_"
			+ std::to_string(static_cast<unsigned long long>(stamp)) + "_" + std::to_string(attempt));
		if (!error && fs::create_directories(root, error) && !error)
			break;
		root.clear();
	}
	if (root.empty())
		return {{"cannot create a temporary directory to check the paths in"}, 1};
	fs::create_directories(root / "tools", error);
	const std::string rootUtf8 = pathToUtf8(root);

	// Give the fake checkout a `tools/<name>`, ours or a stranger's. The two
	// bodies differ only by the marker, which is the whole question isOurCopy()
	// answers -- writing them apart is how a test ends up planting a marked file
	// and asserting the unmarked answer.
	auto plant = [&](std::string_view name, bool ours) {
		const std::string body = ours
			? "# a stand-in, marked " + std::string(kToolMarker) + "\n"
			: std::string("# someone else's script of the same name\n");
		if (std::FILE* handle = openFile(root / "tools" / pathFromUtf8(name), "wb")) {
			std::fwrite(body.data(), 1, body.size(), handle);
			std::fclose(handle);
		}
	};

	for (std::string_view name : kShippedTools) {
		checks += 4;
		const std::string shipped = shippedPath(name);
		if (!fs::is_regular_file(pathFromUtf8(shipped), error))
			problems.push_back(std::string(name) + " is named in a refusal but is not shipped at "
				+ shipped);
		const std::string away = toolPath(name, rootUtf8);
		if (away != shipped)
			problems.push_back(std::string(name) + ": outside a repository the refusal names '"
				+ away + "', not the shipped copy");
		plant(name, true);
		// Not `near`: <windows.h> still defines it as an empty 16-bit macro.
		const std::string inRepo = toolPath(name, rootUtf8);
		if (inRepo != std::string("tools/") + std::string(name))
			problems.push_back(std::string(name) + ": a checkout with its own stand-in is told '"
				+ inRepo + "', not the short path");
		plant(name, false);
		const std::string stranger = toolPath(name, rootUtf8);
		if (stranger != shipped)
			problems.push_back(std::string(name) + ": an unrelated tools/" + std::string(name)
				+ " is handed out as '" + stranger + "' instead of being ignored");
		fs::remove(root / "tools" / pathFromUtf8(name), error);
	}

	// The default base, which is the only one the hook uses.
	checks += 2;
	const std::string restore = gCallerCwd;
	gCallerCwd = rootUtf8;
	{
		const std::string_view name = kShippedTools[0];  // the branch is the same for either name
		if (toolPath(name) != shippedPath(name))
			problems.push_back(std::string(name) + ": with no base given the resolver does not "
				"fall back to the shipped copy");
		plant(name, true);
		if (toolPath(name) != std::string("tools/") + std::string(name))
			problems.push_back(std::string(name) + ": with no base given the resolver does not "
				"read the caller's directory");
		// ... and standalone, where no payload said where the caller is.
		checks += 1;
		const fs::path here = fs::current_path(error);
		gCallerCwd.clear();
		fs::current_path(root, error);
		if (toolPath(name) != std::string("tools/") + std::string(name))
			problems.push_back(std::string(name) + ": with no payload the resolver does not fall "
				"back to the process's own directory");
		fs::current_path(here, error);
	}
	gCallerCwd = restore;
	fs::remove_all(root, error);
	return {problems, checks};
}

int selfTest() {
	int failures = 0;
	const std::vector<SelfTestCase> cases = selfTestCases();
	int checks = static_cast<int>(cases.size());
	const std::pair<std::vector<std::string>, int> groups[] = {checkWiring(), checkPaths()};
	const char* labels[] = {"wiring", "paths"};
	for (size_t group = 0; group < std::size(groups); ++group) {
		checks += groups[group].second;
		for (const std::string& problem : groups[group].first) {
			failures += 1;
			std::fprintf(stderr, "FAIL [%s] %s\n", labels[group], problem.c_str());
		}
	}
	for (const SelfTestCase& test : cases) {
		const bool denied = !scan(test.command, test.shell, true, test.tool).empty();
		if (denied == test.denied)
			continue;
		failures += 1;
		const std::string shown = test.command.size() < 60 ? test.command
			: test.command.substr(0, 57) + "...";
		std::fprintf(stderr, "FAIL [%s/%s] '%s': %s, expected %s\n", test.shell.c_str(),
			test.tool.c_str(), shown.c_str(), denied ? "denied" : "allowed",
			test.denied ? "denied" : "allowed");
	}
	std::printf("%d/%d checks pass\n", checks - failures, checks);
	return failures ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

std::string readAllStdin() {
#ifdef _WIN32
	_setmode(_fileno(stdin), _O_BINARY);
#endif
	std::string out;
	char buffer[8192];
	size_t got = 0;
	while ((got = std::fread(buffer, 1, sizeof(buffer), stdin)) > 0)
		out.append(buffer, got);
	return out;
}

void writeStdout(std::string_view text) {
	std::fwrite(text.data(), 1, text.size(), stdout);
}

constexpr std::string_view kUsage = R"GATE(usage: ask_user_gate [--check COMMAND | --check-file PATH | --self-test]
                     [--shell {bash,powershell}] [--tool NAME]
                     [--platform {auto,windows,posix}]

Refuse shell commands that would stop for a human permission prompt. Reads a
PreToolUse hook payload on stdin unless --check/--check-file/--self-test is
given. Exit 1 when a checked command is denied.

  --check COMMAND    scan one command and print the verdict
  --check-file PATH  scan the command stored in a file (for the multi-line ones
                     an argument cannot carry)
  --shell SHELL      shell to assume (default: bash)
  --tool NAME        tool the command came from; only Monitor differs (its
                     remedy is not the same one) (default: Bash)
  --platform HOST    host the command would run on; only the Git Bash note
                     depends on it (default: auto)
  --self-test        run the built-in scanner, wiring and path-resolver cases
)GATE";

int run(int argc, char** argv) {
	std::optional<std::string> checkCommand;
	std::optional<std::string> checkFile;
	std::string shell = "bash";
	std::string tool = "Bash";
	std::string platform = "auto";
	bool wantSelfTest = false;

	for (int i = 1; i < argc; ++i) {
		std::string argument = argv[i];
		std::string value;
		bool hasInlineValue = false;
		const size_t equals = argument.find('=');
		if (startsWith(argument, "--") && equals != std::string::npos) {
			value = argument.substr(equals + 1);
			argument = argument.substr(0, equals);
			hasInlineValue = true;
		}
		auto takeValue = [&](const char* name) -> bool {
			if (hasInlineValue)
				return true;
			if (i + 1 >= argc) {
				std::fprintf(stderr, "ask_user_gate: %s needs a value\n", name);
				return false;
			}
			value = argv[++i];
			return true;
		};
		if (argument == "--self-test") {
			wantSelfTest = true;
		} else if (argument == "--check") {
			if (!takeValue("--check"))
				return 2;
			checkCommand = value;
		} else if (argument == "--check-file") {
			if (!takeValue("--check-file"))
				return 2;
			checkFile = value;
		} else if (argument == "--shell") {
			if (!takeValue("--shell"))
				return 2;
			if (value != "bash" && value != "powershell") {
				std::fprintf(stderr, "ask_user_gate: --shell must be bash or powershell\n");
				return 2;
			}
			shell = value;
		} else if (argument == "--tool") {
			if (!takeValue("--tool"))
				return 2;
			tool = value;
		} else if (argument == "--platform") {
			if (!takeValue("--platform"))
				return 2;
			if (value != "auto" && value != "windows" && value != "posix") {
				std::fprintf(stderr, "ask_user_gate: --platform must be auto, windows or posix\n");
				return 2;
			}
			platform = value;
		} else if (argument == "-h" || argument == "--help") {
			writeStdout(kUsage);
			return 0;
		} else {
			std::fprintf(stderr, "ask_user_gate: unrecognised argument %s\n", argument.c_str());
			writeStdout(kUsage);
			return 2;
		}
	}

	if (wantSelfTest)
		return selfTest();

	const bool windows = (platform == "auto") ? isWindowsHost() : (platform == "windows");

	std::optional<std::string> command = checkCommand;
	if (checkFile) {
		const std::optional<std::string> content = readFile(pathFromUtf8(*checkFile));
		if (!content) {
			std::fprintf(stderr, "ask_user_gate: cannot read %s\n", checkFile->c_str());
			return 2;
		}
		command = *content;
	}
	if (command) {
		const std::vector<Finding> findings = scan(*command, shell, windows, tool);
		if (findings.empty()) {
			writeStdout("allowed\n");
			return 0;
		}
		writeStdout(render(findings) + "\n");
		return 1;
	}

	// Hook mode. Anything unexpected here must fail OPEN: a broken gate that
	// blocked every command would be worse than the prompts it prevents.
	const std::string payloadText = readAllStdin();
	Json payload;
	if (!JsonParser(payloadText).parse(payload) || payload.type != Json::Type::Object)
		return 0;
	const std::string toolName = payload.memberString("tool_name");
	const std::optional<std::string> payloadShell = shellForTool(toolName);
	if (!payloadShell)
		return 0;
	const std::string cwd = payload.memberString("cwd");
	if (!cwd.empty())
		gCallerCwd = cwd;
	// Monitor's other form is `ws` (a socket, no shell); a missing `command`
	// then means there is nothing to scan, not that something went wrong.
	const Json* toolInput = payload.member("tool_input");
	if (!toolInput || toolInput->type != Json::Type::Object)
		return 0;
	const Json* commandValue = toolInput->member("command");
	if (!commandValue || commandValue->type != Json::Type::String)
		return 0;
	const std::vector<Finding> findings = scan(commandValue->text, *payloadShell, std::nullopt,
		toolName);
	if (findings.empty())
		return 0;

	writeStdout("{\"hookSpecificOutput\": {\"hookEventName\": \"PreToolUse\", "
		"\"permissionDecision\": \"deny\", \"permissionDecisionReason\": \""
		+ jsonEscape(render(findings)) + "\"}}");
	return 0;
}

}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
	_setmode(_fileno(stdout), _O_BINARY);
#endif
	try {
		return run(argc, argv);
	} catch (...) {
		// Same contract as the reference's bare `except`: never turn a bug in
		// the gate into a wall in front of every shell call.
		return 0;
	}
}
