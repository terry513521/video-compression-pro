"""Subprocess execution for external tools.

Single choke point. Every ffmpeg/ffprobe invocation in this package goes through
``run()``, which raises :class:`CommandError` on a non-zero exit instead of returning
a code that a caller may forget to check.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass

from ..errors import CommandError
from ..log import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class CommandResult:
    argv: list[str]
    stdout: str
    stderr: str
    seconds: float


def run(
    argv: list[str],
    *,
    timeout: float | None = None,
    check: bool = True,
    cwd: str | bytes | os.PathLike[str] | None = None,
) -> CommandResult:
    """Run a command to completion, capturing output.

    Args:
        argv: Full argument vector. Never a shell string — no quoting bugs, no injection.
        timeout: Seconds before the process is killed.
        check: Raise :class:`CommandError` on a non-zero exit.
        cwd: Working directory for the child process.

    Raises:
        CommandError: Non-zero exit (when ``check``), timeout, or missing binary.
    """
    log.debug("exec: %s", " ".join(argv))
    started = time.monotonic()
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            cwd=cwd,
        )
    except FileNotFoundError as exc:
        raise CommandError(argv, 127, f"binary not found: {argv[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        stderr = exc.stderr or ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        raise CommandError(argv, 124, f"timed out after {timeout}s\n{stderr}") from exc

    elapsed = time.monotonic() - started
    if proc.stderr and proc.stderr.strip():
        log.debug("stderr (%s): %s", argv[0], proc.stderr.strip()[:4000])
    if check and proc.returncode != 0:
        log.error(
            "command failed rc=%s: %s\n%s",
            proc.returncode,
            " ".join(argv),
            (proc.stderr or proc.stdout or "").strip()[:8000],
        )
        raise CommandError(argv, proc.returncode, proc.stderr or proc.stdout or "")
    return CommandResult(
        argv=argv, stdout=proc.stdout, stderr=proc.stderr, seconds=elapsed
    )
