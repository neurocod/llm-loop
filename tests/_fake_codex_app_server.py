"""A stand-in `codex app-server` for the quota-source tests.

argv: <mode> <pid-log>. Appends its pid to <pid-log> on start, answers
`initialize` and `account/rateLimits/read` over JSONL stdio. `usedPercent` is
the number of reads THIS process has answered, so a test can tell a reused
server (1, 2, ...) from a restarted one (1 again).

Modes: `ok` answers everything; `exit-after-read` exits right after its first
read reply (the next read on it sees EOF); `exit-on-read` exits on its first
read without replying; `hang` never answers a read;
`hang-ignore-eof` never answers a read and outlives stdin EOF by a minute, so
only a kill ends it.
"""

import json
import os
import sys
import time


def main() -> None:
    mode, pid_log = sys.argv[1], sys.argv[2]
    with open(pid_log, "a", encoding="utf-8") as log:
        log.write(f"{os.getpid()}\n")
    reads = 0
    for line in sys.stdin:
        message = json.loads(line)
        method, request_id = message.get("method"), message.get("id")
        if method == "initialize":
            reply = {"id": request_id, "result": {"userAgent": "fake"}}
        elif method == "account/rateLimits/read":
            if mode.startswith("hang"):
                continue
            if mode == "exit-on-read":
                return
            reads += 1
            reply = {"id": request_id, "result": {"rateLimits": {
                "primary": {"usedPercent": reads, "windowDurationMins": 10080},
            }}}
        else:
            continue
        print("not json: a stderr line merged into stdout", flush=True)
        print(json.dumps(reply), flush=True)
        if mode == "exit-after-read" and method == "account/rateLimits/read":
            return
    if mode == "hang-ignore-eof":
        time.sleep(60)


if __name__ == "__main__":
    main()
