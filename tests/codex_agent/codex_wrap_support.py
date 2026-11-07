"""Codex's PowerShell wrapper around a command, for judge tests."""

import shlex


class _CodexWrap:
    """Codex's PowerShell wrapper around a command, quoted the way Codex
    quotes it — POSIX-style (`sandbox.split_shell`) — with `.format` so a
    row reads like the template it replaced."""

    PREFIX = ('"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" '
              "-Command ")

    def format(self, command):
        return self.PREFIX + shlex.quote(command)


_PS = _CodexWrap()
