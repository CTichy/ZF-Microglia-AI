"""
_live_progress.py — surfaces the real progress output MONAI/Cellpose already
produce internally (the same output visible when running these libraries
from a terminal with `tee`) into the plugin's own GUI, for the long,
otherwise-silent calls that only ever showed one static "Running..." label
for their whole duration (do_3D inference, MONAI sliding-window inference).

Why the output was missing in the first place (two different reasons):
  - MONAI's sliding_window_inference(progress=True) draws a raw tqdm bar
    straight to sys.stderr — never captured or shown anywhere in the GUI.
  - Cellpose's do_3D progress goes entirely through Python's `logging`
    module (core_logger/dynamics_logger/models_logger). cellpose/__init__.py
    attaches a NullHandler to its own logger by default, so unless
    something calls cellpose.io.logger_setup() (only the cellpose CLI does
    this) those messages go nowhere at all — not even to a terminal running
    the plugin. This plugin never calls logger_setup(), so cellpose stays
    silent by construction, independent of stdout/stderr redirection.

capture_live_output() covers both: temporarily redirects stdout/stderr and
temporarily attaches a handler to the root logger (propagation is on by
default for cellpose's loggers since logger_setup() is never called here),
forwarding every captured line/log record to a caller-supplied `push(str)`.
"""

import logging
import sys
import threading
from contextlib import contextmanager

# Only one capture can safely own sys.stdout/sys.stderr at a time -- see
# capture_live_output() for what happens if two long operations overlap.
_capture_lock = threading.Lock()


class _LineSink:
    """File-like object: buffers partial writes, pushes complete lines to
    `push` on '\\n' or '\\r' -- tqdm uses '\\r' to redraw its bar in place,
    which we surface as a new appended line instead (a GUI log view can't
    overwrite in place the way a terminal can)."""

    def __init__(self, push):
        self._push = push
        self._buf = ""

    def write(self, s):
        if not s:
            return
        self._buf += s
        while True:
            idx_n = self._buf.find("\n")
            idx_r = self._buf.find("\r")
            candidates = [i for i in (idx_n, idx_r) if i != -1]
            if not candidates:
                return
            idx = min(candidates)
            line, self._buf = self._buf[:idx], self._buf[idx + 1:]
            if line.strip():
                self._push(line)

    def flush(self):
        if self._buf.strip():
            self._push(self._buf.strip())
        self._buf = ""

    def isatty(self):
        return False


class _PushHandler(logging.Handler):
    def __init__(self, push, level=logging.INFO):
        super().__init__(level=level)
        self._push = push

    def emit(self, record):
        try:
            self._push(self.format(record))
        except Exception:
            pass  # a broken push callback must never crash the worker


@contextmanager
def capture_live_output(push):
    """Redirect stdout/stderr and attach a root-logger handler for the
    duration of the `with` block, forwarding every captured line/log
    record to `push(str)`. `push` must be safe to call from a background
    thread (e.g. append-under-lock, matching the progress_cb contract
    every GT-sweep tool in _widget.py already uses).

    If another capture is already active (two long operations started
    concurrently from different tabs), this one is a no-op passthrough
    instead of swapping global stdout/stderr a second time -- correctness
    of the *first* capture's restore matters more than the second
    operation also getting a live log, and that second operation is no
    worse off than the plugin's pre-existing behaviour (silent)."""
    if not _capture_lock.acquire(blocking=False):
        yield
        return

    old_stdout, old_stderr = sys.stdout, sys.stderr
    sink = _LineSink(push)
    handler = _PushHandler(push)
    root = logging.getLogger()
    old_level = root.level
    try:
        sys.stdout = sink
        sys.stderr = sink
        root.addHandler(handler)
        if old_level == logging.NOTSET or old_level > logging.INFO:
            root.setLevel(logging.INFO)
        yield
    finally:
        sink.flush()
        sys.stdout, sys.stderr = old_stdout, old_stderr
        root.removeHandler(handler)
        root.setLevel(old_level)
        _capture_lock.release()
