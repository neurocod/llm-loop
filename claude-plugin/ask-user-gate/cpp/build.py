#!/usr/bin/env python3
"""Configure and build ask_user_gate, dropping the binary next to hooks.json.

One call rather than a configure-then-build chain: the gate this builds refuses
chained shell commands, and a build recipe that has to be typed as one would be
a poor advertisement for it.

  python cpp/build.py                 # Release, incremental
  python cpp/build.py --clean         # throw away the CMake cache first
  python cpp/build.py --config Debug
"""

import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(HERE, "build")
HOOKS_DIR = os.path.normpath(os.path.join(HERE, os.pardir, "hooks"))


def run(command: "list[str]") -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="Release",
                        choices=("Release", "Debug", "RelWithDebInfo",
                                 "MinSizeRel"))
    parser.add_argument("--clean", action="store_true",
                        help="delete the build directory before configuring")
    parser.add_argument("--generator", default=None,
                        help="CMake generator (default: CMake's own choice)")
    parser.add_argument("--no-self-test", action="store_true",
                        help="skip the --self-test run after a successful build")
    options = parser.parse_args()

    if shutil.which("cmake") is None:
        print("cmake is not on PATH. On this machine it ships with Visual "
              "Studio 2022 -- run this from a Developer prompt, or add "
              "'<VS>/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin' "
              "to PATH.", file=sys.stderr)
        return 2

    if options.clean and os.path.isdir(BUILD_DIR):
        shutil.rmtree(BUILD_DIR)

    configure = ["cmake", "-S", HERE, "-B", BUILD_DIR,
                 f"-DCMAKE_BUILD_TYPE={options.config}"]
    if options.generator:
        configure += ["-G", options.generator]
    run(configure)
    # --config is ignored by single-config generators and required by the
    # multi-config one Visual Studio brings, so it is always passed.
    run(["cmake", "--build", BUILD_DIR, "--config", options.config])

    exe = os.path.join(HOOKS_DIR,
                       "ask_user_gate.exe" if os.name == "nt" else "ask_user_gate")
    if not os.path.isfile(exe):
        print(f"build reported success but {exe} is not there", file=sys.stderr)
        return 1
    # Size is the thing the Release flags buy (see CMakeLists.txt), so it is
    # printed on every build rather than measured only when someone suspects it.
    print(f"built {exe} ({os.path.getsize(exe) / 1024:.0f} KiB)")
    if options.no_self_test:
        return 0
    return subprocess.run([exe, "--self-test"]).returncode


if __name__ == "__main__":
    sys.exit(main())
