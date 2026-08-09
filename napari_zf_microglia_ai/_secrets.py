"""
_secrets.py — OS-native encrypted secret storage for the plugin's two
persisted credentials (SMTP app password, LLM API key), via the `keyring`
package: Windows Credential Manager, macOS Keychain, or the Linux Secret
Service API (GNOME Keyring / KWallet), depending on platform. Real
encryption at rest with an OS-managed key, unlike storing these values in
the plugin's own plaintext config.json.

Graceful fallback is not optional here, it's the normal case on some
machines: Windows and macOS always have a working backend, but Linux
needs a running Secret Service daemon with an *unlocked* login keyring.
Confirmed directly on this project's own workstation -- a real Secret
Service backend is detected there, but every call fails with
`KeyringLocked` (no unlocked session available, e.g. over SSH or before
first desktop login) rather than succeeding. So every function here
catches `keyring.errors.KeyringError` (the base class `KeyringLocked`,
`NoKeyringError`, `PasswordSetError`, etc. all inherit from) and degrades
to "value not available" rather than raising -- callers treat that
exactly like "nothing saved yet," not a crash, and the field just needs
re-entering for that session, the same experience as before this module
existed.
"""

SERVICE_NAME = "napari-zf-microglia-ai"


def get_secret(key: str) -> str:
    """Return the stored secret for `key`, or "" if unset or the OS
    credential backend is unavailable/locked. Never raises."""
    try:
        import keyring
        import keyring.errors
        return keyring.get_password(SERVICE_NAME, key) or ""
    except ImportError:
        return ""
    except Exception:
        return ""


def migrate_plaintext_secrets(cfg: dict) -> "tuple[dict, bool]":
    """One-time upgrade path for configs written before secrets moved
    into the OS credential store: if `api_key` or `notify_smtp_password`
    are still sitting in plaintext in a loaded config dict (from an
    older version of this plugin), move them into the OS store and strip
    them from the dict that gets handed back, so they're never written
    to config.json again. Returns (possibly-modified cfg copy, whether
    anything was migrated) -- callers should re-save the config only
    when the second value is True."""
    cfg = dict(cfg)
    migrated = False
    for key in ("api_key", "notify_smtp_password"):
        value = cfg.pop(key, None)
        if value:
            set_secret(key, value)
            migrated = True
    return cfg, migrated


def set_secret(key: str, value: str) -> "str | None":
    """Store `value` under `key` in the OS credential store. An empty
    `value` deletes any existing entry instead (matches clearing a text
    field). Returns None on success, or a short error string on failure
    (no `keyring` installed, or no usable OS backend) -- callers should
    surface this rather than silently losing the value, since it means
    the field will need to be re-entered next session."""
    try:
        import keyring
        import keyring.errors
    except ImportError:
        return "keyring package not installed -- value will not be saved between sessions."
    try:
        if value:
            keyring.set_password(SERVICE_NAME, key, value)
        else:
            try:
                keyring.delete_password(SERVICE_NAME, key)
            except keyring.errors.PasswordDeleteError:
                pass  # nothing was stored -- fine
        return None
    except Exception as exc:
        return (
            f"could not reach the OS credential store ({exc}) -- value will "
            "not be saved between sessions."
        )
