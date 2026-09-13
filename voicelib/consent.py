"""Recorded consent for speech capture, and its invalidation by digest.

S02 requires that consent is created once and then reused; S03 requires that it
stops applying when the thing consented to changes. Neither existed: nothing in
the tree mentioned consent at all, so V25's S02 and S03 arms had no mechanism.

The record lives inside the private XDG layout paths.py already guarantees, is
written 0600 through a temporary file and an atomic rename, and binds a
``subject`` (what was consented to) to a ``digest`` (its exact content at the
time). A digest that no longer matches does not silently re-authorise: consent
to record with one model, or under one notice text, is not consent to record
with another.

Importing this module performs no filesystem work.
"""

from __future__ import annotations

import json
import os
import re
import tempfile

from . import paths

CONSENT_SCHEMA = "kilix.voice.consent/v1"
CONSENT_FILE = "consent.json"
FILE_MODE = 0o600

_SUBJECT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class ConsentError(ValueError):
    """Malformed consent input, or an unreadable record."""


def consent_path() -> str:
    """Return the consent record's path inside the private data directory."""
    return os.path.join(paths.data_dir(), CONSENT_FILE)


def _checked(subject: str, digest: str) -> tuple[str, str]:
    if not isinstance(subject, str) or not _SUBJECT.fullmatch(subject):
        raise ConsentError(
            f"invalid consent subject {subject!r}: expected 1-64 characters "
            "from [A-Za-z0-9._-] starting with a letter or digit.")
    if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
        raise ConsentError(
            f"invalid consent digest {digest!r}: expected 64 lowercase hex "
            "characters (a sha256 of exactly what was consented to).")
    return subject, digest


def _load() -> dict:
    """Return the record on disk, or {} when there is none."""
    try:
        with open(consent_path(), "rb") as handle:
            raw = handle.read()
    except FileNotFoundError:
        return {}
    except OSError as error:
        raise ConsentError(
            f"consent record at {consent_path()} is unreadable ({error}). "
            "Remove it and grant consent again.") from error
    try:
        record = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConsentError(
            f"consent record at {consent_path()} is not valid JSON ({error}). "
            "Remove it and grant consent again.") from error
    if not isinstance(record, dict) or record.get("schema") != CONSENT_SCHEMA:
        raise ConsentError(
            f"consent record at {consent_path()} is not {CONSENT_SCHEMA}. "
            "Remove it and grant consent again.")
    return record


def grant(subject: str, digest: str) -> dict:
    """Record consent for ``subject`` at ``digest``; return the record.

    Idempotent for the same pair: granting twice does not change the recorded
    grant, which is what S02's "reuse" means.
    """
    subject, digest = _checked(subject, digest)
    existing = _load()
    grants = dict(existing.get("grants") or {})
    if grants.get(subject) != digest:
        grants[subject] = digest
    record = {"schema": CONSENT_SCHEMA, "grants": grants}
    _write(record)
    return record


def _write(record: dict) -> None:
    """Write the record 0600 and rename it into place atomically."""
    directory = paths.ensure_private_dir(paths.data_dir())
    handle, temporary = tempfile.mkstemp(dir=directory, prefix=".consent-")
    try:
        with os.fdopen(handle, "wb") as stream:
            # Defence in depth, NOT the guarantor: tempfile.mkstemp already
            # creates 0600 regardless of umask, so removing this line does not
            # change the observed mode -- measured, and the mutation test that
            # deletes it correctly still passes. It is kept so the intended mode
            # is stated at the write site rather than inherited from mkstemp's
            # documented behaviour.
            os.fchmod(stream.fileno(), FILE_MODE)
            stream.write(json.dumps(record, sort_keys=True).encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, consent_path())       # no half-written record
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def granted(subject: str, digest: str) -> bool:
    """Return True only if consent was recorded for this exact digest.

    A record for the subject at a DIFFERENT digest returns False: that is S03's
    invalidation. It is not an error -- the caller's job is to ask again.
    """
    subject, digest = _checked(subject, digest)
    return (_load().get("grants") or {}).get(subject) == digest


def revoke(subject: str) -> bool:
    """Drop any grant for ``subject``; return whether one was present."""
    if not isinstance(subject, str) or not _SUBJECT.fullmatch(subject):
        raise ConsentError(f"invalid consent subject {subject!r}.")
    record = _load()
    grants = dict(record.get("grants") or {})
    if subject not in grants:
        return False
    del grants[subject]
    _write({"schema": CONSENT_SCHEMA, "grants": grants})
    return True
