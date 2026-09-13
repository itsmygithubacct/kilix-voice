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

import fcntl
import hashlib
import json
import os
import re
import tempfile
from contextlib import contextmanager

from . import paths

CONSENT_SCHEMA = "kilix.voice.consent/v1"
CONSENT_FILE = "consent.json"
FILE_MODE = 0o600

_SUBJECT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


LOCK_FILE = "consent.lock"


class ConsentError(ValueError):
    """Malformed consent input, or an unreadable record."""


@contextmanager
def _transaction():
    """Serialise a whole read-modify-write across writers.

    An atomic rename guarantees that readers never see a half-written file. It
    does NOT isolate transactions: two writers that each load, modify their own
    copy and replace the whole file lose one of the two updates. An independent
    review demonstrated exactly that with two concurrent grants of different
    subjects, only one of which survived.

    The lock is a separate file, so it is never the thing being replaced.
    """
    directory = paths.ensure_private_dir(paths.data_dir())
    handle = os.open(os.path.join(directory, LOCK_FILE),
                     os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, FILE_MODE)
    try:
        os.fchmod(handle, FILE_MODE)
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield directory
    finally:
        os.close(handle)          # releases the flock


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
    grants = record.get("grants")
    if grants is not None:
        if not isinstance(grants, dict):
            raise ConsentError(
                f"consent record at {consent_path()} has a malformed 'grants'. "
                "Remove it and grant consent again.")
        for key, value in grants.items():
            # A malformed entry must not read as "no grant for that subject":
            # that would silently downgrade to unconsented and hide tampering.
            if not isinstance(key, str) or not _SUBJECT.fullmatch(key) \
                    or not isinstance(value, str) or not _DIGEST.fullmatch(value):
                raise ConsentError(
                    f"consent record at {consent_path()} has a malformed grant "
                    f"for {key!r}. Remove it and grant consent again.")
    return record


def grant(subject: str, digest: str) -> dict:
    """Record consent for ``subject`` at ``digest``; return the record.

    Idempotent for the same pair: granting twice does not change the recorded
    grant, which is what S02's "reuse" means.
    """
    subject, digest = _checked(subject, digest)
    with _transaction():
        existing = _load()
        grants = dict(existing.get("grants") or {})
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
            # LOAD-BEARING. An earlier annotation here claimed mkstemp gives
            # 0600 "regardless of umask" and that this line was mere defence in
            # depth. That was wrong, and an independent review caught it: a
            # umask can only NARROW creation, so under umask 0777 mkstemp
            # yields mode 0000 and only this fchmod restores 0600. Measured:
            # umask 0000 -> 0600, 0077 -> 0600, 0777 -> 0000.
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
    with _transaction():
        record = _load()
        grants = dict(record.get("grants") or {})
        if subject not in grants:
            return False
        del grants[subject]
        _write({"schema": CONSENT_SCHEMA, "grants": grants})
    return True


def capture_digest(model_id: str, engine: str, payload_digest: str = "") -> str:
    """Return the digest a dictation consent is bound to.

    S03: what the consent is FOR must be in the digest, so changing any of it
    invalidates the grant rather than silently carrying it over. Recognising
    with a different model, or with the same model whose payload changed
    underneath, is not what the user agreed to.

    ``payload_digest`` is the installed artefact's own identity when a caller
    can supply one. It is deliberately a REQUIRED part of the input rather than
    an optional extra: an empty value is recorded as empty and still changes the
    digest the moment a real one appears, so consent granted before artefact
    identity was available does not survive its arrival.
    """
    for name, value in (("model_id", model_id), ("engine", engine),
                        ("payload_digest", payload_digest)):
        if not isinstance(value, str):
            raise ConsentError(
                f"{name} must be a string, got {type(value).__name__}.")
    material = "\x00".join((CONSENT_SCHEMA, model_id, engine, payload_digest))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()
