"""Stall watchdog for long evaluation processes.

A harness process occasionally wedges inside the simulator (SAPIEN/Vulkan or
curobo) with the CPU spinning and no further output; it then holds a GPU slot
for hours without producing an episode.  This module lets the process kill
itself after a period without progress, so that the shard shows a non-zero exit
code so a supervisor can restart it.

Call :func:`beat` whenever real progress happens (a command executed, a model
reply received, an episode started) and :func:`start` once at startup.
"""

from __future__ import annotations

import faulthandler
import os
import sys
import threading
import time

_last = time.time()
_lock = threading.Lock()
_note = "startup"
_hard_seconds = 0.0


def beat(note: str = "") -> None:
    """Record that progress happened just now."""
    global _last, _note
    with _lock:
        _last = time.time()
        if note:
            _note = note
    if _hard_seconds:
        # Rearm the C-level timer too: a hang inside a native extension (curobo, SAPIEN) can hold the GIL
        # forever, and then the Python watchdog thread below never gets to run.  faulthandler's timer does
        # not need the GIL, so it is the one that actually kills such a process.
        faulthandler.dump_traceback_later(_hard_seconds, exit=True)


def start(stall_seconds: float = 900.0, poll_seconds: float = 30.0, exit_code: int = 3) -> threading.Thread:
    """Abort the process when no :func:`beat` arrives for ``stall_seconds``.

    Two timers, because one is not enough: the thread below reports which step stalled, and a
    ``faulthandler`` timer (rearmed by every :func:`beat`) kills the process even when a native
    extension holds the GIL and no Python code can run at all.
    """
    global _hard_seconds
    if stall_seconds > 0:
        _hard_seconds = stall_seconds * 2      # the reporting thread gets the first chance
        faulthandler.dump_traceback_later(_hard_seconds, exit=True)

    def loop() -> None:
        while True:
            time.sleep(poll_seconds)
            with _lock:
                idle = time.time() - _last
                note = _note
            if idle > stall_seconds:
                sys.stderr.write(
                    f"WATCHDOG: no progress for {idle:.0f}s (last: {note}); dumping stacks and aborting\n")
                sys.stderr.flush()
                try:
                    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
                except Exception:  # noqa: BLE001
                    pass
                sys.stderr.flush()
                os._exit(exit_code)

    thread = threading.Thread(target=loop, name="harness-watchdog", daemon=True)
    thread.start()
    return thread
