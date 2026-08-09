"""
_secrets.py — layered secret storage for the plugin's two persisted
credentials (SMTP app password, LLM API key).

Tier 1 (preferred): the OS-native encrypted credential store via the
`keyring` package -- Windows Credential Manager, macOS Keychain, or the
Linux Secret Service API (GNOME Keyring / KWallet). Real encryption at
rest with an OS-managed key.

Tier 2 (fallback, only when Tier 1 is unavailable): a local file
encrypted with Fernet/AES (via `cryptography`, already a transitive
dependency of keyring's Linux backend, declared directly here so it's
guaranteed present on every platform), keyed by a machine-local key
generated once and stored alongside it in the plugin's own config
directory. This is honestly a *weaker* guarantee than Tier 1 -- the key
lives right next to the ciphertext it protects, on the same machine and
under the same user account, so it defends against casual exposure (an
accidental `cat config.json`, a stray backup of just the JSON file, a
screen-share) rather than a determined attacker with full read access to
this user's home directory. It's used anyway, deliberately, because Tier
1 genuinely doesn't work on some real machines this plugin runs on: SSH-
only Linux sessions never get the PAM-driven keyring auto-unlock that a
graphical login provides, confirmed directly on this project's own
workstation (a real Secret Service backend is detected, but every call
fails with `KeyringLocked` -- no unlocked session, not a missing
backend). Prompting an interactive master password every session was
considered and rejected -- friction the whole point of persisting these
values was meant to remove -- so Tier 2 needs no prompt at all. It's
still strictly better than the plaintext config.json this project used
before either tier existed.

If BOTH tiers fail (Tier 2 needs `cryptography`, present by default, but
theoretically missing, or the config directory being unwritable), every
function degrades to "value not available" rather than raising --
callers treat that exactly like "nothing saved yet," and the field just
needs re-entering for that session.
"""

import json
from pathlib import Path

SERVICE_NAME = "napari-zf-microglia-ai"

_FALLBACK_DIR = Path.home() / ".config" / "napari-zf-microglia-ai"
_FALLBACK_KEY_PATH = _FALLBACK_DIR / ".secret_key"
_FALLBACK_STORE_PATH = _FALLBACK_DIR / ".secrets_encrypted"


def _chmod_owner_only(path: Path) -> None:
    """Best-effort -- a no-op on Windows (NTFS ACLs work differently and
    the OS credential store is reliably available there anyway, so this
    fallback tier matters far less on that platform)."""
    try:
        path.chmod(0o600)
    except Exception:
        pass


def _get_fernet():
    from cryptography.fernet import Fernet
    _FALLBACK_DIR.mkdir(parents=True, exist_ok=True)
    if _FALLBACK_KEY_PATH.exists():
        key = _FALLBACK_KEY_PATH.read_bytes()
    else:
        key = Fernet.generate_key()
        _FALLBACK_KEY_PATH.write_bytes(key)
        _chmod_owner_only(_FALLBACK_KEY_PATH)
    return Fernet(key)


def _fallback_read_all() -> dict:
    if not _FALLBACK_STORE_PATH.exists():
        return {}
    try:
        fernet = _get_fernet()
        return json.loads(fernet.decrypt(_FALLBACK_STORE_PATH.read_bytes()).decode())
    except Exception:
        return {}


def _fallback_get(key: str) -> str:
    try:
        return _fallback_read_all().get(key, "")
    except Exception:
        return ""


def _fallback_set(key: str, value: str) -> None:
    """Raises on failure -- callers decide what "both tiers failed"
    means for their error message."""
    fernet = _get_fernet()
    data = _fallback_read_all()
    if value:
        data[key] = value
    else:
        data.pop(key, None)
    _FALLBACK_STORE_PATH.write_bytes(fernet.encrypt(json.dumps(data).encode()))
    _chmod_owner_only(_FALLBACK_STORE_PATH)


def get_secret(key: str) -> str:
    """Return the stored secret for `key` (Tier 1, then Tier 2), or ""
    if unset in both or neither is usable. Never raises."""
    try:
        import keyring
        value = keyring.get_password(SERVICE_NAME, key)
        if value:
            return value
    except Exception:
        pass
    return _fallback_get(key)


def migrate_plaintext_secrets(cfg: dict) -> "tuple[dict, bool]":
    """One-time upgrade path for configs written before secrets moved out
    of config.json: if `api_key` or `notify_smtp_password` are still
    sitting in plaintext in a loaded config dict (from an older version
    of this plugin), move them into layered storage and strip them from
    the dict that gets handed back, so they're never written to
    config.json again. Returns (possibly-modified cfg copy, whether
    anything was migrated) -- callers should re-save the config only when
    the second value is True."""
    cfg = dict(cfg)
    migrated = False
    for key in ("api_key", "notify_smtp_password"):
        value = cfg.pop(key, None)
        if value:
            set_secret(key, value)
            migrated = True
    return cfg, migrated


def set_secret(key: str, value: str) -> "str | None":
    """Store `value` under `key` -- tries the OS credential store first
    (Tier 1), falls back to the local encrypted file (Tier 2) if that's
    unavailable. An empty `value` deletes any existing entry from
    whichever tier(s) have one, instead of storing an empty string.

    Returns None whenever the value ends up persisted *somewhere*
    (Tier 1 or Tier 2 -- both are "fine, nothing for the caller to warn
    about"), or a short error string only when neither tier could store
    it, meaning the value truly won't survive to next session. Callers
    already print this to the console rather than blocking on it."""
    try:
        import keyring
        import keyring.errors
        if value:
            keyring.set_password(SERVICE_NAME, key, value)
        else:
            try:
                keyring.delete_password(SERVICE_NAME, key)
            except keyring.errors.PasswordDeleteError:
                pass  # nothing was stored there -- fine
        # Tier 1 worked -- don't also keep a redundant Tier 2 copy lying
        # around; if a stale one exists from an earlier locked session,
        # clear it now that the real store is reachable again.
        try:
            _fallback_set(key, "")
        except Exception:
            pass
        return None
    except Exception:
        pass

    try:
        _fallback_set(key, value)
        print(
            f"{key}: OS credential store unavailable -- saved to a local "
            f"encrypted file instead ({_FALLBACK_STORE_PATH}). Still "
            "encrypted, but the key protecting it lives on this same "
            "machine, so it's a weaker guarantee than the OS store."
        )
        return None
    except Exception as exc:
        return (
            f"could not save anywhere (OS credential store unavailable; local "
            f"encrypted-file fallback also failed: {exc}) -- value will not "
            "be saved between sessions."
        )
