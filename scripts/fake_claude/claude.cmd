@echo off
rem A .cmd shim like npm's own claude.cmd, so the runner meets the same
rem cmd.exe-in-front-of-the-CLI process tree it meets in a real run.
python "%~dp0fake_claude.py" %*
