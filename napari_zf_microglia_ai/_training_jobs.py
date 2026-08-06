"""
_training_jobs.py — cross-platform detached process management for
long-running training jobs (MONAI, Cellpose-SAM).

Both train.py (hours-days) and train_xzyz.py (~20h) need to keep running
after napari closes, and the GUI needs to be able to find, poll, and stop
them again later — including after a full napari restart. The obvious
Linux/Mac answer is tmux, but this plugin ships a native Windows install
path too, and tmux does not exist there. So instead of a tmux-on-Linux /
something-else-on-Windows split, everything here is built on primitives
that behave the same on all three platforms: a detached subprocess.Popen
+ PID tracking via psutil, with stdout/stderr redirected to a plugin-
controlled log file that the GUI tails. The only thing this gives up
versus tmux is the ability to `tmux attach` from an external terminal on
Linux — not a real loss, since the GUI's own log-tail view shows the same
content, and `tail -f <log_path>` still works manually if wanted.
"""

import platform
import subprocess

import psutil


def launch_detached(argv, cwd, log_path, conda_env):
    """
    Launch `argv` inside `conda run -n <conda_env> --no-capture-output`,
    detached so it survives this process (napari) exiting. stdout+stderr
    are redirected to log_path. Returns the PID of the launched process.
    """
    full_argv = ["conda", "run", "-n", conda_env, "--no-capture-output", *argv]
    kwargs = {}
    if platform.system() == "Windows":
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        )
    else:
        kwargs["start_new_session"] = True  # setsid() — detach from our session

    log_f = open(log_path, "wb")
    try:
        proc = subprocess.Popen(
            full_argv, cwd=str(cwd), stdout=log_f, stderr=subprocess.STDOUT, **kwargs
        )
    finally:
        log_f.close()
    return proc.pid


def is_running(pid) -> bool:
    """True if `pid` still exists and isn't a zombie."""
    if not pid:
        return False
    try:
        return psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def kill_process_tree(pid, timeout=5) -> None:
    """
    Terminate `pid` and all its descendants (conda run spawns a child
    python process, so killing only the top PID would leave training
    running). Graceful SIGTERM/terminate first, force-kill any survivors.
    """
    if not pid:
        return
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    procs = parent.children(recursive=True) + [parent]
    for p in procs:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass

    _, alive = psutil.wait_procs(procs, timeout=timeout)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass


def tail_log(log_path, n_bytes=8192) -> str:
    """Return the last `n_bytes` of log_path, decoded, without ever reading
    a multi-day log file from the start — bounded regardless of file size."""
    try:
        with open(log_path, "rb") as f:
            f.seek(0, 2)  # end
            size = f.tell()
            f.seek(max(0, size - n_bytes))
            data = f.read()
        text = data.decode("utf-8", errors="replace")
        if size > n_bytes:
            text = "... (truncated) ...\n" + text
        return text
    except FileNotFoundError:
        return "(log file not created yet)"
    except Exception as exc:
        return f"(could not read log: {exc})"


def run_subprocess_job(argv, cwd, conda_env) -> subprocess.CompletedProcess:
    """
    Blocking run of `argv` inside `conda run -n <conda_env>`, for fast
    (minutes-scale) jobs like prepare_data.py — meant to be called from
    inside a background threading.Thread, not detached (no need to
    survive napari closing for a job this short; the caller thread
    already keeps it off the Qt main thread).
    """
    full_argv = ["conda", "run", "-n", conda_env, "--no-capture-output", *argv]
    return subprocess.run(
        full_argv, cwd=str(cwd), capture_output=True, text=True,
    )
