"""agent-tools/cpu_load.py: rates from two raw samples, and the JSON door,
without launching CIM -- so the suite runs on every platform."""
import collections
import io
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import cpu_load

TICKS = 10_000_000  # 1 s in Timestamp_Sys100NS units
# The caller-named `--orphan-image`s of the orphan cases, in `image_key` form.
IMAGES = frozenset({"node", "chrome"})


def sample(ts, procs):
    """One raw sample: (name, pid, cpu ticks) rows stamped `ts`."""
    return [{"Name": n, "IDProcess": pid, "PercentProcessorTime": cpu, "Timestamp_Sys100NS": ts}
            for n, pid, cpu in procs]


# The first sample's stamp: a UTC FILETIME, 1_762_000_000_000 ms after the Unix epoch.
T0 = cpu_load.FILETIME_EPOCH + 1_762_000_000_000 * 10_000
# What `--json` names as the span of `raw`'s two samples, 1 s apart.
SPAN = {"fromMs": 1_762_000_000_000, "toMs": 1_762_000_001_000}


def raw(first, second, cpus=4, self_pid=999):
    return {"cpus": cpus, "self": self_pid,
            "first": sample(T0, first), "second": sample(T0 + TICKS, second)}


def run_tool(monkeypatch, capsys, argv, *, data=None, error=None, stderr="", code=0):
    monkeypatch.setattr(sys, "argv", ["cpu_load.py", *argv])
    monkeypatch.setattr(cpu_load, "WINDOWS", True)

    def query(*args, **kwargs):
        assert kwargs["timeout"] == cpu_load.QUERY_TIMEOUT_S
        if error:
            raise error
        return SimpleNamespace(returncode=code, stdout=data, stderr=stderr)

    monkeypatch.setattr(cpu_load.subprocess, "run", query)
    result = cpu_load.main()
    return result, capsys.readouterr()


def test_rates_are_deltas_per_pid_over_the_window():
    # 4 CPUs over 1 s: idle 3 core-seconds -> 25 % load; node used 1.5 cores.
    data = raw([("Idle", 0, 0), ("_Total", 0, 0), ("node#4", 7, 5 * TICKS), ("server", 8, 0)],
               [("Idle", 0, 3 * TICKS), ("_Total", 0, 9 * TICKS), ("node#2", 7, 6.5 * TICKS),
                ("server", 8, 0.4 * TICKS)])
    load, procs = cpu_load.all_rates(data, data["cpus"])
    assert load == 25
    # `node#4` and `node#2` are one process (pid 7) whose slot was renumbered.
    assert sorted(procs, key=lambda p: p["IDProcess"]) == [
        {"Name": "node", "IDProcess": 7, "PercentProcessorTime": 150},
        {"Name": "server", "IDProcess": 8, "PercentProcessorTime": 40},
    ]


def test_rates_drop_the_query_itself_reused_pids_and_anomalies():
    data = raw([("Idle", 0, 0), ("WmiPrvSE#1", 3, 0), ("powershell", 999, 0), ("python", 5, 0),
                ("node", 6, 0), ("chrome", 9, 0), ("gone", 10, 0)],
               [("Idle", 0, 4 * TICKS), ("WmiPrvSE#1", 3, 3 * TICKS), ("powershell", 999, 2 * TICKS),
                ("python", 5, TICKS), ("cl", 6, TICKS), ("chrome", 9, 401 * TICKS),
                ("fresh", 11, TICKS)])
    load, procs = cpu_load.all_rates(data, data["cpus"])
    procs = cpu_load.rates(procs, {999, 5})
    assert load == 0
    # WmiPrvSE (the provider), $PID and the caller, a pid reused under another
    # name, a rate above 100 x 4 CPUs, and pids in only one sample are all out.
    assert procs == []


def test_total_and_idle_are_never_ranked():
    # Both carry pid 0 in a live query; a plausible Idle rate (3 of 4 cores)
    # would rank if `before` kept it.
    data = raw([("_Total", 0, 0), ("Idle", 0, 0)], [("_Total", 0, 4 * TICKS), ("Idle", 0, 3 * TICKS)])
    load, procs = cpu_load.all_rates(data, data["cpus"])
    assert load == 25
    assert procs == []


def test_rates_without_an_idle_pair_have_no_load():
    load, procs = cpu_load.all_rates({"first": sample(TICKS, [("node", 1, 0)]),
                                    "second": sample(2 * TICKS, [("node", 1, TICKS // 2)])}, 2)
    assert load is None
    assert procs == [{"Name": "node", "IDProcess": 1, "PercentProcessorTime": 50}]


def test_json_selection_matches_table(monkeypatch, capsys):
    data = json.dumps(raw(
        [("Idle", 0, 0), ("node", 1, 0), ("server", 2, 0), ("other", 3, 0)],
        [("Idle", 0, 3 * TICKS), ("node", 1, 0.8 * TICKS), ("server", 2, 2.4 * TICKS),
         ("other", 3, 3 * TICKS)], cpus=8))
    argv = ["--top", "1", "--match", "NODE|server"]
    code, output = run_tool(monkeypatch, capsys, [*argv, "--json"], data=data)
    assert code == 0 and not output.err
    assert json.loads(output.out) == {"load": 62, "procs": [
        {"Name": "server", "IDProcess": 2, "PercentProcessorTime": 240},
    ], **SPAN}
    code, output = run_tool(monkeypatch, capsys, argv, data=data)
    assert code == 0
    assert "CPU load: 62 %" in output.out and "server" in output.out
    assert "node" not in output.out and "other" not in output.out


def test_the_callers_own_pid_is_excluded(monkeypatch, capsys):
    me = os.getpid()
    data = json.dumps(raw([("Idle", 0, 0), ("python", me, 0)],
                          [("Idle", 0, 4 * TICKS), ("python", me, TICKS)]))
    code, output = run_tool(monkeypatch, capsys, ["--json"], data=data)
    assert code == 0
    assert json.loads(output.out) == {"load": 0, "procs": [], **SPAN}


def test_the_span_is_the_idle_rows_stamps_in_unix_ms():
    # A judged window counts a sample by the moments it saw, not by when it was asked.
    rows = [("Idle", 0, 0), ("node", 1, 0)]
    data = raw(rows, rows)
    data["second"][0]["Timestamp_Sys100NS"] = T0 + 7_000_000  # 0.7 s later; node's row keeps T0 + 1 s
    assert cpu_load.interval_ms(data) == (1_762_000_000_000, 1_762_000_000_700)
    assert cpu_load.interval_ms({"first": [], "second": sample(T0, rows)}) is None


def with_parents(data, parents, started=None):
    """`data` with `CreatingProcessID` set from `parents` (pid -> parent pid) in
    both samples, and `ElapsedTime` (the raw start, a FILETIME) from `started`
    (pid -> start), by default the pid itself: a parent with a lower pid started first."""
    started = started or {}
    for rows in (data["first"], data["second"]):
        for p in rows:
            p["CreatingProcessID"] = parents.get(p["IDProcess"], 1)
            p["ElapsedTime"] = started.get(p["IDProcess"], p["IDProcess"])
    return data


def test_own_tree_refuses_a_parent_that_started_after_its_child():
    # A reused pid: our server 12 (started at 50) holds the pid a foreign
    # runner 30 (started at 10) was created by, long dead; 30's child 31 with
    # it. A recycled pid must not adopt them — their load and their server would
    # be hidden from the wait as "ours".
    rows = [("Idle", 0, 0), ("node", 10, 0), ("server", 12, 0), ("node", 30, 0), ("server", 31, 0)]
    data = with_parents(raw(rows, rows), {12: 10, 30: 12, 31: 30}, {10: 5, 12: 50, 30: 10, 31: 40})
    assert cpu_load.own_tree(data, 10) == {10, 12}
    # The same tree with honest start times adopts them: the comparison is the whole guard.
    data = with_parents(raw(rows, rows), {12: 10, 30: 12, 31: 30}, {10: 5, 12: 50, 30: 60, 31: 70})
    assert cpu_load.own_tree(data, 10) == {10, 12, 30, 31}


def test_own_tree_takes_no_link_it_cannot_verify_nor_one_through_the_sampler():
    rows = [("Idle", 0, 0), ("node", 10, 0), ("chrome", 11, 0), ("python", 999, 0), ("server", 40, 0)]
    data = with_parents(raw(rows, rows), {11: 10, 999: 10, 40: 999}, {40: 2000})
    # The sampler (999, here the query's own pid) is ours but parents nothing:
    # a stranger naming its pid is a stale link.
    assert cpu_load.own_tree(data, 10, frozenset({999})) == {10, 11, 999}
    assert cpu_load.own_tree(data, 10) == {10, 11, 999, 40}
    # A row without a start time (an older query, a counter that did not answer) links nothing.
    for rows_ in (data["first"], data["second"]):
        for p in rows_:
            if p["IDProcess"] == 11:
                del p["ElapsedTime"]
    assert cpu_load.own_tree(data, 10, frozenset({999})) == {10, 999}


def test_own_tree_is_the_root_and_its_descendants():
    # 10 started 11 and 12, 12 started 13; 20 is a stranger; Idle's pid 0 is
    # nobody's parent (a 0 link would adopt the whole machine).
    data = with_parents(raw([("Idle", 0, 0), ("node", 10, 0), ("chrome", 11, 0), ("server", 12, 0),
                             ("python", 13, 0), ("server", 20, 0)],
                            [("Idle", 0, 0), ("node", 10, 0), ("chrome", 11, 0), ("server", 12, 0),
                             ("python", 13, 0), ("server", 20, 0)]),
                        {10: 4, 11: 10, 12: 10, 13: 12, 20: 0})
    assert cpu_load.own_tree(data, 10) == {10, 11, 12, 13}
    assert cpu_load.own_tree(data, 99) == {99}


@pytest.mark.parametrize("creator", [99, 11], ids=["dead-creator", "recycled-creator"])
def test_orphans_started_during_the_run_and_their_children_are_unattributed(creator):
    # 20 may be our Chrome, left behind by a cmd/npm wrapper absent from both
    # samples. Creator 99 is dead; creator 11 is an own pid reused AFTER 20
    # started. Its surviving child 21 cannot establish an external lineage.
    rows = [("node", 10, 0), ("node", 11, 0), ("chrome", 20, 0), ("chrome", 21, 0),
            ("cl", 30, 0), ("cl", 31, 0)]
    data = with_parents(raw(rows, rows), {11: 10, 20: creator, 21: 20, 30: 98, 31: 30},
                        {10: 100, 11: 200, 20: 150, 21: 160, 30: 50, 31: 170})
    uncertain = set()
    assert cpu_load.own_tree(data, 10, unattributed=uncertain, orphan_images=IMAGES) == {10, 11}
    assert uncertain == {20, 21}
    # 30 predates the run, so both it and its NEW child 31 remain external.
    assert not uncertain.intersection({30, 31})


@pytest.mark.parametrize("missing", [10, 20], ids=["missing-root-start", "missing-child-start"])
def test_missing_start_times_cannot_establish_an_external_process(missing):
    rows = [("node", 10, 0), ("chrome", 20, 0)]
    data = with_parents(raw(rows, rows), {20: 99}, {10: 100, 20: 50})
    for part in (data["first"], data["second"]):
        for row in part:
            if row["IDProcess"] == missing:
                del row["ElapsedTime"]
    uncertain = set()
    assert cpu_load.own_tree(data, 10, unattributed=uncertain, orphan_images=IMAGES) == {10}
    assert uncertain == {20}


@pytest.mark.parametrize("image", ["MSBuild.exe", "mspdbsrv", "vctip"])
def test_orphans_of_images_not_named_and_their_children_remain_external(image):
    rows = [("node", 10, 0), (image, 20, 0), ("cl", 21, 0), ("link", 22, 0)]
    after = [(name, pid, TICKS) for name, pid, _ in rows]
    data = with_parents(raw(rows, after), {20: 99, 21: 20, 22: 20},
                        {10: 100, 20: 150, 21: 160, 22: 170})
    uncertain = set()
    tree = cpu_load.own_tree(data, 10, unattributed=uncertain, orphan_images=IMAGES)
    assert tree == {10}
    assert uncertain == set()
    _, procs = cpu_load.all_rates(data, 4)
    assert cpu_load.others_share(procs, tree | uncertain, 4) == 75


@pytest.mark.parametrize("image", ["node.exe", "NoDe#4"])
def test_a_named_image_matches_by_base_name_case_and_exe_aside(image):
    rows = [("node", 10, 0), (image, 20, 0), ("chrome", 21, 0)]
    data = with_parents(raw(rows, rows), {20: 99, 21: 20}, {10: 100, 20: 150, 21: 160})
    uncertain = set()
    assert cpu_load.own_tree(data, 10, unattributed=uncertain, orphan_images=IMAGES) == {10}
    assert uncertain == {20, 21}


def test_without_named_images_no_orphan_is_unattributed():
    """The default: the script cannot know which images a caller's run starts."""
    rows = [("node", 10, 0), ("node", 20, 0), ("chrome", 21, 0)]
    data = with_parents(raw(rows, rows), {20: 99, 21: 20}, {10: 100, 20: 150, 21: 160})
    uncertain = set()
    assert cpu_load.own_tree(data, 10, unattributed=uncertain) == {10}
    assert uncertain == set()


def test_orphan_cpu_is_reported_separately_from_positive_external_cpu(monkeypatch, capsys):
    # The orphan tree used 40 % and a known external, pre-run process used
    # another 40 %. A BUSY decision must see that uncertainty even though the
    # positively external share is already above its ordinary load threshold.
    names = [("Idle", 0), ("node", 10), ("chrome", 20), ("chrome", 21), ("cl", 30)]
    data = with_parents(raw([(name, pid, 0) for name, pid in names],
                           [(name, pid, ticks * TICKS) for (name, pid), ticks in
                            zip(names, [0.4, 0.4, 0.8, 0.8, 1.6])]),
                        {20: 99, 21: 20, 30: 98}, {10: 100, 20: 150, 21: 160, 30: 50})
    code, output = run_tool(monkeypatch, capsys, ["--json", "--own-tree", "10", "--orphan-image", "Chrome.exe"],
                            data=json.dumps(data))
    assert code == 0 and not output.err
    answer = json.loads(output.out)
    assert (answer["load"], answer["own"], answer["others"], answer["unattributed"]) == (90, 10, 40, 40)
    assert [p["IDProcess"] for p in answer["procs"]] == [30]


def test_own_tree_is_reported_and_not_ranked_and_find_skips_it(monkeypatch, capsys):
    # 4 CPUs, 1 s: idle 1 core-second -> 75 % load. The tree (10, 11) used 2
    # cores = 50 % of the machine; the stranger 20 used 1 core.
    data = json.dumps(with_parents(raw(
        [("Idle", 0, 0), ("node", 10, 0), ("chrome", 11, 0), ("server", 12, 0), ("server", 20, 0)],
        [("Idle", 0, TICKS), ("node", 10, 0.5 * TICKS), ("chrome", 11, 1.5 * TICKS),
         ("server", 12, 0), ("server", 20, TICKS)]), {11: 10, 12: 10, 20: 3}, {20: 3}))
    code, output = run_tool(monkeypatch, capsys, ["--json", "--own-tree", "10", "--find", "SERVER"],
                            data=data)
    assert code == 0 and not output.err
    assert json.loads(output.out) == {
        "load": 75, "own": 50, "others": 25, "unattributed": 0,
        "procs": [{"Name": "server", "IDProcess": 20, "PercentProcessorTime": 100}],
        # The tree's own server (12, idle) is not found; a stranger's is, at any CPU.
        "found": [{"Name": "server", "IDProcess": 20}],
        **SPAN,
    }
    # Whole-name match: `serv` alone finds nothing.
    code, output = run_tool(monkeypatch, capsys, ["--json", "--find", "serv"], data=data)
    assert json.loads(output.out)["found"] == []


def test_own_counts_the_sampling_itself_though_it_is_never_ranked(monkeypatch, capsys):
    # 4 CPUs, 1 s, load 100 %: the tree's node 10 used 1 core; the sampler's
    # python (this process, a child of 10), its PowerShell (999, the query's
    # $PID, a child of the python) and the WMI provider (3, started by a service
    # host) used 1 core between them; the stranger 20 used 2. Others are 50 %:
    # the sampling is ours, not the stranger's.
    me = os.getpid()
    rows0 = [("Idle", 0, 0), ("node", 10, 0), ("python", me, 0), ("powershell", 999, 0),
             ("WmiPrvSE", 3, 0), ("cl", 20, 0)]
    rows1 = [("Idle", 0, 0), ("node", 10, TICKS), ("python", me, 0.25 * TICKS),
             ("powershell", 999, 0.5 * TICKS), ("WmiPrvSE", 3, 0.25 * TICKS), ("cl", 20, 2 * TICKS)]
    data = json.dumps(with_parents(raw(rows0, rows1), {me: 10, 999: me, 3: 1, 20: 2},
                                   {10: 5, me: 6, 999: 7, 3: 1, 20: 2}))
    code, output = run_tool(monkeypatch, capsys, ["--json", "--own-tree", "10"], data=data)
    assert code == 0 and not output.err
    answer = json.loads(output.out)
    assert answer["load"] == 100 and answer["own"] == 50 and answer["others"] == 50
    # ...and the ranking still shows none of the sampling.
    assert [p["IDProcess"] for p in answer["procs"]] == [20]


def test_others_is_only_what_is_known_to_be_outside_the_tree(monkeypatch, capsys):
    # 4 CPUs, 1 s, load 100 %: the tree's node 10 used 1 core; its child 11,
    # spawned inside the gap (only in the second sample), used 2 -- in the load,
    # rateless; the stranger 20 used 1. `load - own` would be 75 % "others";
    # positively attributed, the stranger's 25 % is all there is.
    rows0 = [("Idle", 0, 0), ("node", 10, 0), ("cl", 20, 0)]
    rows1 = [("Idle", 0, 0), ("node", 10, TICKS), ("server", 11, 2 * TICKS), ("cl", 20, TICKS)]
    data = json.dumps(with_parents(raw(rows0, rows1), {11: 10, 20: 2}, {20: 2}))
    code, output = run_tool(monkeypatch, capsys, ["--json", "--own-tree", "10"], data=data)
    assert code == 0 and not output.err
    answer = json.loads(output.out)
    assert (answer["load"], answer["own"], answer["others"]) == (100, 25, 25)


@pytest.mark.parametrize("error", [FileNotFoundError("powershell missing"),
    subprocess.TimeoutExpired("powershell", 15)])
def test_query_failure_is_stderr(monkeypatch, capsys, error):
    code, output = run_tool(monkeypatch, capsys, ["--json"], error=error)
    assert code == 1 and not output.out
    assert "CPU query failed" in output.err


@pytest.mark.parametrize("data,stderr,code,reason", [
    ("", "CIM refused", 1, "CIM refused"),
    ("invalid", "", 0, "invalid JSON"),
    ('{"cpus": 4, "first": [{"Name": "node", "IDProcess": 1}], "second": [{"Name": "node", "IDProcess": 1}]}',
     "", 0, "invalid JSON"),
    (json.dumps({"cpus": 4, "first": [], "second": []}), "", 0, "no usable Idle"),
])
def test_bad_query_output(monkeypatch, capsys, data, stderr, code, reason):
    result, output = run_tool(monkeypatch, capsys, ["--json"], data=data, stderr=stderr, code=code)
    assert result == 1 and not output.out
    assert reason in output.err


class FakeStreamQuery:
    """The `--stream` PowerShell: `lines` on stdout, then EOF."""

    def __init__(self, lines, stderr=""):
        self.stdout = iter(lines)
        self.stderr = io.StringIO(stderr)
        self.killed = False
        self.reaped = False

    def kill(self):
        self.killed = True

    def wait(self):
        assert self.killed
        self.reaped = True
        return 0


def snapshot_line(ts, procs, self_pid=999):
    return json.dumps({"cpus": 4, "self": self_pid, "rows": sample(ts, procs)}) + "\n"


def run_stream(monkeypatch, capsys, argv, query):
    monkeypatch.setattr(sys, "argv", ["cpu_load.py", *argv])
    monkeypatch.setattr(cpu_load, "WINDOWS", True)
    started = []

    def popen(command, **kwargs):
        started.append(command)
        return query

    monkeypatch.setattr(cpu_load.subprocess, "Popen", popen)
    return cpu_load.main(), capsys.readouterr(), started


# A judged window needs samples that tile it, not one per call.
def test_stream_answers_every_consecutive_pair_with_spans_that_tile(monkeypatch, capsys):
    rows = [[("Idle", 0, 0), ("cl", 20, 0)],
            [("Idle", 0, 2 * TICKS), ("cl", 20, 2 * TICKS)],
            [("Idle", 0, 3 * TICKS), ("cl", 20, 5 * TICKS)]]
    query = FakeStreamQuery([snapshot_line(T0 + i * TICKS, r) for i, r in enumerate(rows)])
    code, output, started = run_stream(monkeypatch, capsys, ["--json", "--stream", "--own-tree", "10"], query)
    assert code == 1 and "CPU query failed" in output.err  # PowerShell ended: the stream did
    assert query.killed
    pairs = [json.loads(line) for line in output.out.splitlines()]
    assert [(p["fromMs"], p["toMs"]) for p in pairs] == [
        (SPAN["fromMs"], SPAN["toMs"]), (SPAN["toMs"], SPAN["toMs"] + 1_000)]
    # 4 CPUs: idle 2 then 1 core-second of 4 -> 50 %, 75 %; cl outside the tree 2 then 3 cores.
    assert [(p["load"], p["others"]) for p in pairs] == [(50, 50), (75, 75)]
    assert started[0][:4] == ["powershell", "-NoProfile", "-NonInteractive", "-Command"]
    assert f"Get-Process -Id {os.getpid()}" in started[0][4]


# Windows reuses pids, so the loop's "caller alive"
# test is the pid AND its start time when the loop began.
def test_stream_query_runs_while_the_callers_pid_keeps_its_start_time():
    query = cpu_load.stream_query(4242)
    alive = "(Get-Process -Id 4242 -ErrorAction SilentlyContinue).StartTime"
    assert f"$born = {alive}; " in query
    assert f"while ($born -and {alive} -eq $born) {{ " in query


# Both schedules project the counters through ONE
# fragment, so a column added for one is there for the other.
def test_one_shot_and_stream_take_the_same_snapshot_fragment():
    for query in (cpu_load.QUERY, cpu_load.stream_query(4242)):
        assert query.startswith(cpu_load.TAKE)
        assert query.count("Get-CimInstance") == 1


TOOLS = Path(cpu_load.__file__).resolve().parent
# A Python child's whole life in the two pins below: 0.08 / 0.09 s measured
# 2026-11-07 (16-33 % CPU). Far over it, for a Python start under a
# neighbour's build; only a red (a hung stream) waits it out.
STREAM_CHILD_S = 20

# A `--stream` whose PowerShell answers pair after pair, forever, as fast as
# the reader takes them; its own stdout is the test's pipe.
ENDLESS_STREAM = r"""
import io, itertools, json, sys
import cpu_load
T0, TICKS = cpu_load.FILETIME_EPOCH + 1_762_000_000_000 * 10_000, 10_000_000
class Query:
    stdout = (json.dumps({"cpus": 4, "self": 999, "rows": [{"Name": "Idle", "IDProcess": 0,
        "PercentProcessorTime": i * TICKS, "Timestamp_Sys100NS": T0 + i * TICKS}]}) + "\n"
        for i in itertools.count())
    stderr = io.StringIO()
    def kill(self): pass
    def wait(self): return 0
cpu_load.subprocess.Popen = lambda *a, **kw: Query()
cpu_load.WINDOWS = True  # not sys.platform: see cpu_load.WINDOWS
sys.argv = ["cpu_load.py", "--json", "--stream"]
sys.exit(cpu_load.main())
"""


# The reader going away exits 0, not with the 120 of a
# failed exit-time flush — a real pipe whose reader closed, not a mocked print.
def test_stream_exits_0_when_its_reader_closes_the_pipe():
    child = subprocess.Popen([sys.executable, "-c", ENDLESS_STREAM], cwd=TOOLS,
                             env={**os.environ, "PYTHONPATH": str(TOOLS)},
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        # A thread, so a child hanging before its first line fails within
        # STREAM_CHILD_S instead of hanging the job.
        lines = []
        reader = threading.Thread(target=lambda: lines.append(child.stdout.readline()),
                                  daemon=True)
        reader.start()
        reader.join(STREAM_CHILD_S)
        first = lines[0] if lines else b""
        if not first:
            # A child that died (or hung) before its first line: the red must
            # carry its exit code and traceback (seen once on ubuntu-latest
            # 3.12 CI 2026-11-08, empty readline, not reproduced in 300 WSL
            # runs). communicate drains stderr while it waits.
            if not lines:
                child.kill()
            _, err = child.communicate(timeout=STREAM_CHILD_S)
            pytest.fail(f"stream child printed nothing{'' if lines else ' in time'}, "
                        f"exit {child.returncode}:\n{err.decode(errors='replace')}")
        assert json.loads(first)["load"] == 75, first
        child.stdout.close()
        child.wait(timeout=STREAM_CHILD_S)
        err = child.stderr.read().decode(errors="replace")
        assert child.returncode == 0, err
        assert "Exception ignored" not in err
    finally:
        child.kill()
        child.wait()


# A PowerShell writing errors faster than anyone
# reads them blocks in its write; `--stream` drains its stderr all along.
def test_stream_drains_powershells_stderr_and_reports_its_tail(monkeypatch, capsys):
    lines = [snapshot_line(T0, [("Idle", 0, 0)]), snapshot_line(T0 + TICKS, [("Idle", 0, TICKS)])]
    # Far over any pipe buffer, then the answers, then EOF.
    script = ("import sys\n"
              "for i in range(20_000): sys.stderr.write(f'Get-CimInstance : Access denied {i}\\n')\n"
              "sys.stderr.flush()\n"
              f"sys.stdout.write({''.join(lines)!r})\n")
    real_popen = subprocess.Popen
    started = []

    def popen(command, **kwargs):
        started.append(real_popen([sys.executable, "-c", script], **kwargs))
        return started[-1]

    monkeypatch.setattr(sys, "argv", ["cpu_load.py", "--json", "--stream"])
    monkeypatch.setattr(cpu_load, "WINDOWS", True)
    monkeypatch.setattr(cpu_load.subprocess, "Popen", popen)
    result = []
    runner = threading.Thread(target=lambda: result.append(cpu_load.main()), daemon=True)
    runner.start()
    runner.join(STREAM_CHILD_S)
    if runner.is_alive():
        started[0].kill()
        pytest.fail("--stream blocked: PowerShell's stderr was not drained")
    output = capsys.readouterr()
    assert result == [1] and len(output.out.splitlines()) == 1
    said = output.err.split("CPU query failed:\n", 1)[1].splitlines()
    assert said == [f"Get-CimInstance : Access denied {i}" for i in range(20_000 - cpu_load.STDERR_TAIL_LINES, 20_000)]


@pytest.mark.parametrize("exit_path", ["broken-pipe", "invalid-json", "invalid-pair", "eof"])
def test_stream_reaps_child_and_stops_reader_on_every_exit(monkeypatch, capsys, exit_path):
    lines = [snapshot_line(T0, [("Idle", 0, 0)]), snapshot_line(T0 + TICKS, [("Idle", 0, TICKS)])]
    if exit_path == "invalid-json":
        lines = ["invalid\n"]
    elif exit_path == "invalid-pair":
        lines = [snapshot_line(T0, []), snapshot_line(T0 + TICKS, [])]
    query = FakeStreamQuery(lines)
    release = threading.Event()
    readers = []
    real_thread = threading.Thread

    class HeldStderr:
        def readline(self, size):
            # Hold EOF until join: a missing join must fail even if the OS
            # would otherwise schedule the reader before stream() returns.
            release.wait(STREAM_CHILD_S)
            return ""

    class Reader(real_thread):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            readers.append(self)

        def join(self, timeout=None):
            assert query.killed and query.reaped
            assert timeout is not None and 0 < timeout <= cpu_load.STDERR_JOIN_S
            release.set()
            super().join(timeout)

    def broken_pipe(*args, **kwargs):
        raise BrokenPipeError()

    query.stderr = HeldStderr()
    monkeypatch.setattr(cpu_load.threading, "Thread", Reader)
    if exit_path == "broken-pipe":
        monkeypatch.setattr(cpu_load, "print", broken_pipe, raising=False)
        monkeypatch.setattr(cpu_load, "_stdout_to_devnull", lambda: None)
    try:
        code, _, _ = run_stream(monkeypatch, capsys, ["--json", "--stream"], query)
        assert code == (0 if exit_path == "broken-pipe" else 1)
        assert query.killed and query.reaped
        assert len(readers) == 1 and not readers[0].is_alive()
    finally:
        release.set()
        for reader in readers:
            real_thread.join(reader, STREAM_CHILD_S)


def test_stream_does_not_pass_off_a_query_error_as_a_closed_reader(monkeypatch, capsys):
    """Only a failed write to stdout means the reader left. An OSError from
    PowerShell's pipe once exited 0 with an empty stderr, like a closed
    reader: a suspect for the silent child of ubuntu CI 2026-11-08."""
    def failing_pipe():
        yield snapshot_line(T0, [("Idle", 0, 0)])
        raise OSError("pipe read failed")

    query = FakeStreamQuery([])
    query.stdout = failing_pipe()
    monkeypatch.setattr(cpu_load, "_stdout_to_devnull", lambda: None)
    with pytest.raises(OSError, match="pipe read failed"):
        run_stream(monkeypatch, capsys, ["--json", "--stream"], query)
    assert query.killed and query.reaped


@pytest.mark.parametrize("completed", [False, True], ids=["unterminated", "completed-lines"])
def test_stderr_retention_is_bounded_even_before_a_newline(completed):
    tail = collections.deque(maxlen=cpu_load.STDERR_TAIL_LINES)
    huge_line = "x" * (cpu_load.STDERR_TAIL_CHARS * 16) + "last diagnostic"
    text = (huge_line + "\n") * cpu_load.STDERR_TAIL_LINES if completed else huge_line

    def check_retained():
        assert sum(map(len, tail)) <= cpu_load.STDERR_TAIL_CHARS
        assert all(len(line) <= cpu_load.STDERR_LINE_CHARS for line in tail)

    class CheckedPipe(io.StringIO):
        def readline(self, size=-1):
            # Inspect between reads, while a huge line is still unfinished.
            check_retained()
            assert 0 < size <= cpu_load.STDERR_READ_CHARS
            return super().readline(size)

    cpu_load._drain(CheckedPipe(text), tail)
    check_retained()
    assert tail[-1] == huge_line[-cpu_load.STDERR_LINE_CHARS:]


def test_stream_needs_json(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["cpu_load.py", "--stream"])
    assert cpu_load.main() == 2
    assert "--stream needs --json" in capsys.readouterr().err


def test_non_windows_reports_reason_without_spawning(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["cpu_load.py", "--json"])
    monkeypatch.setattr(cpu_load, "WINDOWS", False)
    monkeypatch.setattr(cpu_load.subprocess, "run", lambda *a, **kw: pytest.fail("spawned"))
    assert cpu_load.main() == 1
    assert "requires Windows" in capsys.readouterr().err
