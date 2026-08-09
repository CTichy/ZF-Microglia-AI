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
import re
import subprocess
from pathlib import Path

import psutil

# Per-script model-selection-metric config for patience-based early stopping
# (see patience_exceeded()). Each log line format is completely different
# between the two training scripts, but the counting/comparison logic that
# consumes these values is the same function for both -- this is the only
# place the two scripts' early-stopping behavior actually differs.
#
# Both patterns capture (epoch, metric_value) as groups 1/2, so the "best
# checkpoint" can always be reported/located by its real training epoch,
# not just its position in the log. MONAI's epoch and Dice value are on
# separate lines ("Epoch N Summary:" ... "Full-brain Dice: X [MODEL
# SELECTION]"), hence DOTALL to span them; Cellpose-SAM has both on one
# line ("N, train_loss=..., test_loss=X, ...").
MONAI_METRIC = dict(
    pattern=r"Epoch (\d+) Summary:.*?Full-brain Dice:\s*([\d.]+)\s*\[MODEL SELECTION\]",
    flags=re.DOTALL,
    higher_is_better=True,
    label="Full-brain Dice",
)
CELLPOSE_METRIC = dict(
    pattern=r"(\d+), train_loss=[\d.]+, test_loss=([\d.]+)",
    flags=0,
    higher_is_better=False,
    label="test_loss",
)


# Generated verbatim into a standalone .py file when `notify` is used (see
# launch_detached) -- stdlib only (subprocess/smtplib/email/re), no
# dependency on this package being importable, since it has to keep
# running under whatever conda_env the training script itself uses. All
# variable values are embedded via repr() (not string interpolation of
# raw text), so paths containing spaces/quotes round-trip safely as valid
# Python literals; the template body itself uses %-formatting rather than
# f-strings/braces so it survives being passed through str.format().
_NOTIFY_SUPERVISOR_TEMPLATE = '''\
import re, smtplib, subprocess, sys
from email.mime.text import MIMEText

argv = {argv!r}
cwd = {cwd!r}
log_path = {log_path!r}
conda_env = {conda_env!r}
to_addr = {to_addr!r}
smtp_host = {smtp_host!r}
smtp_port = {smtp_port!r}
smtp_user = {smtp_user!r}
smtp_password = {smtp_password!r}
job_label = {job_label!r}
pattern = {pattern!r}
flags = {flags!r}
higher_is_better = {higher_is_better!r}
metric_label = {metric_label!r}

full_argv = ["conda", "run", "-n", conda_env, "--no-capture-output"] + argv
with open(log_path, "wb") as log_f:
    proc = subprocess.run(full_argv, cwd=cwd, stdout=log_f, stderr=subprocess.STDOUT)

try:
    text = open(log_path, "r", errors="replace").read()
except Exception:
    text = ""
matches = list(re.finditer(pattern, text, flags))
if matches:
    series = [(int(m.group(1)), float(m.group(2))) for m in matches]
    best = max(series, key=lambda t: t[1]) if higher_is_better else min(series, key=lambda t: t[1])
    metric_line = "Best %s = %.4f at epoch %d (of %d checkpoints logged)." % (
        metric_label, best[1], best[0], len(series))
else:
    metric_line = "No checkpoints were logged -- check the log file for errors."

status_word = "finished" if proc.returncode == 0 else ("stopped (exit code %s)" % proc.returncode)
subject = "[napari-zf-microglia-ai] %s training %s" % (job_label, status_word)
body = (
    "%s training has stopped.\\n\\n"
    "Status: %s\\n"
    "%s\\n\\n"
    "Log file: %s\\n"
) % (job_label, status_word, metric_line, log_path)

try:
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = to_addr
    with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=30) as server:
        server.login(smtp_user, smtp_password)
        server.sendmail(smtp_user, [to_addr], msg.as_string())
except Exception as exc:
    sys.stderr.write("email notification failed: %s\\n" % exc)
'''


def launch_detached(argv, cwd, log_path, conda_env, notify=None):
    """
    Launch `argv` inside `conda run -n <conda_env> --no-capture-output`,
    detached so it survives this process (napari) exiting. stdout+stderr
    are redirected to log_path. Returns the PID of the launched process.

    If `notify` is given -- dict with to_addr/smtp_host/smtp_port/
    smtp_user/smtp_password/job_label/metric_cfg -- the training command
    is wrapped in a small self-contained supervisor script that becomes
    the detached process instead of `argv` directly: the supervisor runs
    the real command (still writing the same log_path the GUI tails
    live), waits for it to exit, then emails a completion report. Because
    the supervisor itself is what's detached, the email still gets sent
    even if napari closes and never reopens before the job finishes --
    the same "survives napari closing" guarantee as the training run
    itself, extended to the notification.

    kill_process_tree(pid) needs no changes for this: the real training
    process is a descendant of the supervisor (conda run -> supervisor.py
    -> conda run -> training script), and kill_process_tree already walks
    all descendants recursively -- killing the tree kills the supervisor
    too, before it reaches the email step, so an explicit Stop Training
    click correctly sends no "stopped" email.
    """
    if notify:
        supervisor_path = Path(f"{log_path}.supervisor.py")
        metric_cfg = notify["metric_cfg"]
        supervisor_path.write_text(_NOTIFY_SUPERVISOR_TEMPLATE.format(
            argv=list(argv), cwd=str(cwd), log_path=str(log_path), conda_env=conda_env,
            to_addr=notify["to_addr"], smtp_host=notify["smtp_host"], smtp_port=notify["smtp_port"],
            smtp_user=notify["smtp_user"], smtp_password=notify["smtp_password"],
            job_label=notify["job_label"], pattern=metric_cfg["pattern"],
            flags=int(metric_cfg.get("flags", 0)), higher_is_better=metric_cfg["higher_is_better"],
            metric_label=metric_cfg["label"],
        ))
        supervisor_log = Path(f"{log_path}.supervisor_log")
        return _popen_detached(["python", str(supervisor_path)], cwd, supervisor_log, conda_env)
    return _popen_detached(argv, cwd, log_path, conda_env)


def _popen_detached(argv, cwd, log_path, conda_env):
    """Shared by launch_detached's two paths (direct, and notify-wrapped
    via the supervisor script) -- see launch_detached's docstring."""
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


def send_notification_email(to_addr, smtp_host, smtp_port, smtp_user, smtp_password,
                             subject, body):
    """Send a plain-text notification email directly via SMTP_SSL, for
    long operations that run in-process (a background thread inside this
    still-open napari process — Tab 1 Run, Tab 2 Cellpose-SAM Segmentation,
    Tab 5's Cellprob/Large-contact and Best-Epoch sweeps), as opposed to
    Tab 4's detached training launches, which need the standalone
    supervisor-script approach above since they must keep running (and
    therefore be able to email) even after napari itself has closed. Only
    a "napari must stay open" version is needed here since these
    operations already don't survive napari closing either way.

    Returns None on success, or a short error string on failure — never
    raises, so a broken email config can't crash the operation whose
    completion it was reporting."""
    try:
        from email.mime.text import MIMEText
        import smtplib
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = to_addr
        with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=30) as server:
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, [to_addr], msg.as_string())
        return None
    except Exception as exc:
        return str(exc)


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


def _parse_metric_series(log_path, metric_cfg):
    """Return [(epoch, value), ...] for every checkpoint matched by
    metric_cfg['pattern'] (capture groups 1=epoch, 2=value) in log_path,
    in the order they were printed."""
    try:
        with open(log_path, "r", errors="replace") as f:
            text = f.read()
    except FileNotFoundError:
        return []
    return [
        (int(m.group(1)), float(m.group(2)))
        for m in re.finditer(metric_cfg["pattern"], text, metric_cfg.get("flags", 0))
    ]


def patience_exceeded(log_path, metric_cfg, patience):
    """
    Patience-based early-stop check, shared verbatim by both MONAI and
    Cellpose-SAM training (see MONAI_METRIC/CELLPOSE_METRIC) -- the only
    thing that differs per script is which metric/regex/direction is
    passed in, not the stopping rule itself: `patience` is a count of
    checkpoints (not epochs), matching the plugin's own checkpoint
    interval, since that's the only cadence the GUI can observe from the
    log regardless of how each script defines its internal epoch/val
    cadence.

    Also doubles as a plain "what's the best checkpoint so far" query --
    callers that only want best_epoch/best_value (e.g. once a job has
    stopped on its own, not via early-stop) can call this with any
    `patience` and just ignore `exceeded`.

    Returns a dict: {exceeded, n_checkpoints, best_value, best_epoch,
    checkpoints_since_best}. `patience <= 0` means exceeded is always
    False (disabled) -- checked here so every caller gets the same
    "best so far" info regardless of whether early-stop is enabled.
    """
    series = _parse_metric_series(log_path, metric_cfg)
    if not series:
        return dict(exceeded=False, n_checkpoints=0, best_value=None,
                     best_epoch=None, checkpoints_since_best=0)

    values = [v for _, v in series]
    if metric_cfg["higher_is_better"]:
        best_index = max(range(len(values)), key=lambda i: values[i])
    else:
        best_index = min(range(len(values)), key=lambda i: values[i])

    checkpoints_since_best = len(values) - 1 - best_index
    exceeded = patience > 0 and checkpoints_since_best >= patience
    best_epoch, best_value = series[best_index]
    return dict(
        exceeded=exceeded, n_checkpoints=len(values), best_value=best_value,
        best_epoch=best_epoch, checkpoints_since_best=checkpoints_since_best,
    )


def write_best_checkpoint_pointer(models_dir, model_name, best_epoch, suffix="best_recommended"):
    """
    Write a small text pointer file '{model_name}_{suffix}.txt' naming the
    recommended checkpoint, so it doesn't require reading the log/GUI
    history to identify later. Only meaningful for Cellpose-SAM
    (train_xzyz.py saves only periodic epoch checkpoints, no separate
    best-tracking) -- MONAI's train.py already auto-saves its own
    best_model_fullstack.pth, so the GUI never needs to call this for
    MONAI jobs.

    Deliberately a plain text pointer, not a copy and not an OS symlink --
    checkpoints are large (100s of MB) so copying wastes disk, and
    os.symlink needs elevated privileges/Developer Mode on Windows (a hard
    link would dodge that, but silently fails across filesystem/drive
    boundaries). A one-line text file naming the target resolves
    identically on every platform with zero special permissions.

    Returns the pointer file's Path, or raises FileNotFoundError if the
    target checkpoint (e.g. deleted, or not a save_every multiple) isn't
    on disk -- a stale pointer is never written.
    """
    models_dir = Path(models_dir)
    target_name = f"{model_name}_epoch_{best_epoch:04d}"
    if not (models_dir / target_name).exists():
        raise FileNotFoundError(f"Best checkpoint not found on disk: {models_dir / target_name}")
    pointer = models_dir / f"{model_name}_{suffix}.txt"
    pointer.write_text(target_name + "\n")
    return pointer


def read_best_checkpoint_pointer(models_dir, model_name, suffix="best_recommended"):
    """
    Resolve a pointer written by write_best_checkpoint_pointer() back to
    the actual checkpoint path. Returns None if no pointer exists yet, or
    if the checkpoint it names is no longer on disk.
    """
    models_dir = Path(models_dir)
    pointer = models_dir / f"{model_name}_{suffix}.txt"
    if not pointer.exists():
        return None
    target = models_dir / pointer.read_text().strip()
    return target if target.exists() else None


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
