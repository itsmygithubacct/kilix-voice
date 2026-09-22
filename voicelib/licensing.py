"""The weights gate: no model weight is fetched without a covering receipt.

OS-V-VERIFY F2 found that the only speech-model install action the operating
system image advertises reached a download with no acceptance of any kind. The
catalog row runs ``kilix stt --install <model> --default <model>``; that
reached :func:`install_model`, which handed the terminal straight to the pinned
installer, which fetched the archive. Nobody had seen the licence, and on the
unattended provisioning path nobody was there to see it.

OD-S and OD-BB require that weights are acquired only after the user has seen
the licence and accepted it. This module is the precondition. Every route in
this checkout that can cause weights to be fetched calls
:func:`require_covering_receipt` first, and refuses when it raises.

**The library is not weights.** ``libvosk.so`` is Apache-2.0 code, not a model.
Loading it, probing for it, reporting on it and installing it need no receipt,
and nothing here is on those paths. Only the model-weight install actions are
gated. See ``README.md`` for the one coupling this cannot reach.

Where the answer comes from
---------------------------
kilix-license is the single licence authority (OD-AJ). It owns the records, the
verbatim first-use screen, agreement capture and the receipts. This module does
not copy any of that: it asks the authority, through the authority's own public
API, and refuses when the authority is not there to ask. Copying the records
here would make kilix-voice a second authority that drifts from the first.

**A receipt is never shipped.** The only thing that may produce a receipt is an
acceptance the user performed on that user's own machine, in kilix-content's
first-use flow. A receipt must never be vendored into a repository, baked into
an image, provisioned onto a machine, or written by a build. Doing so would
satisfy this gate on every machine at once while nobody had seen a licence --
the OS-V-VERIFY F2 hole restored in a form that reads as compliant, and a
direct contradiction of OD-S ("the user ... gives an explicit acceptance before
download"). Shipping a receipt is not an acceptable remedy for a machine that
cannot install weights; the acceptable outcomes are that the user accepts at
first use, or that the weights are not installed. If a build ever needs a
build-time attestation, OQ-C4 already rules that it must be a distinct schema
and never a ``kilix.license.receipt/v1``.

That rule is necessary because of what the receipt schema does **not** bind: it
carries no subject, no timestamp and no signature, so any well-formed receipt
in the store covers forever, for any user, on any machine. Making a receipt
bind the person and the moment is kilix-license's to do -- it owns the schema
(OD-AJ) -- and is tracked there, not here. This gate verifies what the
authority defines; it cannot bind more than the authority binds.

This module never writes to the receipt store, on any path. It only reads one
that already exists: see :func:`_opened_store`.

Importing this module performs no filesystem, subprocess, network or authority
work. Everything happens inside the call.
"""

from __future__ import annotations

import os
import pathlib

from . import models, paths

# The authority's Python distribution. Resolved by ordinary import, so the
# deployment that installs kilix-license decides where it lives; this module
# adds no search path of its own, and so no new way to point the licence
# authority at somebody else's code.
AUTHORITY_DISTRIBUTION = "kilix-license"
AUTHORITY_MODULE = "kilix_license"

# The names this checkout calls. An importable module that does not expose all
# of them is not the authority, whatever it is called.
AUTHORITY_API = (
    "AssetRef", "CoverageRefused", "ReceiptStore", "RecordIndex",
    "covers", "load_determined_records", "require",
)

# Receipts are per user and shared by every consumer of the authority, so they
# live beside the rest of the stack's writable state rather than under this
# one component. Where exactly is the authority's to say (OD-AJ, V-ACC-VERIFY
# F7): kilix-license names the root in `kilix_license.receipt_store_root()`,
# with one override, `$KILIX_LICENSE_RECEIPTS`, and the writer -- kilix-content's
# first-use flow reached as `kilix models install` -- files there. This gate
# reads the root from the same function, so the two cannot part company.
AUTHORITY_ENV_RECEIPTS = "KILIX_LICENSE_RECEIPTS"
# kilix-voice's own override, from before the authority named a root. Kept as
# a legacy alias so a deployment that set it keeps reading where it pointed,
# but it moves only this reader: the writer never reads it. When the
# authority's own variable is set as well, the authority's answer wins, so
# this gate reads where receipts are filed rather than where the alias points.
ENV_RECEIPTS = "KILIX_VOICE_LICENSE_RECEIPTS"
RECEIPTS_LEAF = "license-receipts"

# A licence refusal is not an installer fault, and an unattended caller has to
# be able to tell the two apart: provisioning that sees this status knows a
# human has to accept a licence, and that retrying will not help.
LICENCE_REFUSED_EXIT = 3


# The one acceptance route in the stack is kilix-content's first-use flow
# (`kilix_content.first_use.install_with_agreement`): it renders the licence
# screen, captures the agreement, writes the receipt and only then fetches.
# It is reached as `kilix models install <asset id>`.
ACCEPT_COMMAND = "kilix models install"

# kilix-content files an asset under its UPSTREAM id, which is not this
# catalog's id for the two Vosk models. Naming this catalog's id would send the
# user to an asset that does not exist, so the refusal names the content id.
#
# The two sides are bound together by the licence record, not by either id: at
# kilix-content 7543aa30 each asset below names exactly the record digest this
# authority resolves for the catalog id beside it, so a receipt written by that
# flow is the receipt this gate finds. Verified digests are in V-ACC-IMPL.md.
# If an upstream id moves, this table moves with it; the digests are the check.
CONTENT_ASSET_ID = {
    "small-en-us": "vosk-model-small-en-us-0.15",
    "lgraph-en-us": "vosk-model-en-us-0.22-lgraph",
    "piper-en-us-kristin-medium": "piper-en-us-kristin-medium",
    "vibevoice-asr-bitnet": "vibevoice-asr-bitnet",
}


# The refusal's second line: the probe that re-checks the same question and
# fetches nothing in either branch. It is an addition to the acceptance route
# above, never a replacement for it -- a probe reports, it does not obtain
# consent -- and it is useful in the window before `kilix models install`
# reaches the first-use flow. Each command answers for its own catalog, so the
# line must name the command that actually accepts this id: the Piper voice is
# kilix-tts's only downloadable model, every other gated id is kilix-stt's.
CHECK_FLAG = "--check-licence"
CHECK_TOOL_BY_MODEL = {models.PIPER_KRISTIN_MODEL: "kilix-tts"}
DEFAULT_CHECK_TOOL = "kilix-stt"


def content_asset_id(catalog_id: str) -> str:
    """Return the id kilix-content files this model's weights under."""
    return CONTENT_ASSET_ID.get(catalog_id, catalog_id)


def accept_command(catalog_id: str) -> str:
    """Return the first-use command that shows the licence and records consent."""
    return f"{ACCEPT_COMMAND} {content_asset_id(catalog_id)}"


def check_command(catalog_id: str) -> str:
    """Return the probe that re-checks this model, naming the tool that owns it."""
    tool = CHECK_TOOL_BY_MODEL.get(catalog_id, DEFAULT_CHECK_TOOL)
    return f"{tool} {CHECK_FLAG} {catalog_id}"


class LicenseRefused(RuntimeError):
    """Weights may not be fetched: no receipt covers this model."""

    def __init__(self, catalog_id: str, reason: str) -> None:
        self.catalog_id = catalog_id
        self.reason = reason
        super().__init__(
            f"refusing to fetch the {catalog_id} model weights: {reason}. "
            "Model weights are fetched only after their licence has been "
            f"shown and accepted.\nRun: {accept_command(catalog_id)}\n"
            f"Re-check with: {check_command(catalog_id)}")


class AuthorityUnavailable(LicenseRefused):
    """The licence authority is not installed, so nothing can be verified."""


def receipt_store_root() -> str:
    """Return the directory the licence authority publishes receipts into.

    The authority's own answer, ``kilix_license.receipt_store_root()``: the
    root its writers file acceptances in. Composing it here instead is what
    V-ACC-VERIFY F7 found -- a writer honouring ``$KILIX_LICENSE_RECEIPTS``
    and a reader that did not, so an acceptance the user gave was filed where
    this gate never looked. ``$KILIX_VOICE_LICENSE_RECEIPTS`` is still read,
    as a legacy alias (see :data:`ENV_RECEIPTS`), only while the authority's
    variable is unset. An authority too old to name a root leaves the
    composition this module always used, ``$GPU_TERMINAL_HOME/license-receipts``
    -- which is what such an authority's writers were given too.

    It may import the authority, and so is called only inside a call, never at
    import time. Any error the authority raises (a relative root, for one)
    propagates; :func:`require_covering_receipt` turns it into a refusal.
    """
    legacy = os.environ.get(ENV_RECEIPTS)
    if legacy and not os.environ.get(AUTHORITY_ENV_RECEIPTS):
        return os.path.abspath(os.path.expanduser(legacy))
    try:
        import kilix_license
        named = kilix_license.receipt_store_root
    except Exception:
        named = None
    if named is None:
        return os.path.join(paths.gpu_terminal_home(), RECEIPTS_LEAF)
    return str(named())


def authority(catalog_id: str):
    """Return the kilix-license module, or refuse.

    Absent authority is a refusal, never a pass. A machine that cannot check a
    licence is a machine that must not fetch weights; the alternative -- fetch
    when the checker is missing -- is the unattended hole F2 reported, wearing
    a different hat.
    """
    try:
        import kilix_license
    except ImportError as error:
        raise AuthorityUnavailable(
            catalog_id,
            f"the {AUTHORITY_DISTRIBUTION} authority is not installed, so no "
            "licence receipt can be verified") from error
    except Exception as error:
        # An authority that raises anything else while importing -- a broken
        # install, a syntax error, a module that raises on purpose -- must
        # refuse the same way, with the licence status and a message. Letting
        # it out would exit 1 with a traceback, which says "the installer
        # broke" when the truth is "no licence could be checked", and inside
        # the curses screen it would take the terminal down instead of
        # printing. It already failed closed; this makes it refuse cleanly.
        raise AuthorityUnavailable(
            catalog_id,
            f"the {AUTHORITY_DISTRIBUTION} authority could not be loaded "
            f"({_detail(error)}), so no licence receipt can be verified"
        ) from error
    missing = [name for name in AUTHORITY_API if not hasattr(kilix_license, name)]
    if missing:
        raise AuthorityUnavailable(
            catalog_id,
            f"the importable {AUTHORITY_MODULE} does not expose the licence "
            f"authority's API (missing {', '.join(sorted(missing))})")
    return kilix_license


def _opened_store(kilix_license, root: str):
    """Return the authority's receipt store, opened without creating it.

    ``ReceiptStore.__init__`` is a *writer's* constructor: it does
    ``mkdir(parents=True, exist_ok=True)`` and then ``chmod(root, 0o700)``, so
    merely constructing it to ask a question changes the filesystem. A refusal
    would leave a store it had re-moded (``0755`` -> ``0700``), and a store at
    ``0000`` would be repaired to ``0700`` and the install would then proceed
    on the strength of a repair this command performed. kilix-voice verifies
    receipts and never produces them, so it must not do either.

    This subclass keeps every reading method the authority defines -- the glob,
    the name-for-digest rule, the regular-file/size discipline, the parse --
    and replaces only the constructor's store-creation. The reading logic is
    still the authority's, so there is no second implementation to drift
    (OD-AJ). Removing the creation from ``ReceiptStore`` itself would be the
    tidier fix, but that class is kilix-license's and this repository is not
    its owner; what this repository can do is not trigger it, and that is what
    this does.

    Every caller wraps the reads below in ``except Exception -> refuse``, so an
    authority whose future ``__init__`` sets state this bypasses fails closed.
    """

    class _OpenedReceiptStore(kilix_license.ReceiptStore):
        """The authority's store, opened read-only: never created, never re-moded."""

        def __init__(self, root: str) -> None:  # noqa: D107 - see the parent
            self.root = pathlib.Path(root)

    return _OpenedReceiptStore(root)


def require_covering_receipt(catalog_id: str, *, manifest_digest: str | None = None):
    """Return the receipt that covers ``catalog_id``, or raise LicenseRefused.

    Nothing is written anywhere, on the refusing path or on the covered one.
    No directory under the model store is created, opened for writing, or
    touched at all; the receipt store is only read, and only if it already
    exists -- it is never created, and its mode is never changed (see
    :func:`_opened_store`).

    ``manifest_digest`` is the OD-AI binding to the exact payload. kilix-voice
    does not fetch the payload and does not know its manifest -- the pinned
    installer owns both -- so the ordinary call omits it and the check is
    scoped to the licence record instead: the authority must hold a receipt
    for this model's record, and ``covers()`` must accept it against that
    record (licence id, licensor, licence text digest, every agreement-bound
    condition, the decision class, and the record digest itself). A caller
    that does know the manifest passes it, and gets the full binding-scoped
    ``require()``. See V-ACC-IMPL.md for what that difference leaves open.
    """
    kilix_license = authority(catalog_id)
    try:
        records = kilix_license.RecordIndex(
            kilix_license.load_determined_records())
    except Exception as error:
        # Same reasoning as the import above: an authority that cannot produce
        # its records refuses with the licence status, not a traceback.
        raise AuthorityUnavailable(
            catalog_id,
            f"the {AUTHORITY_DISTRIBUTION} authority could not read its "
            f"licence records ({_detail(error)})") from error
    try:
        record = records.by_id(catalog_id)
    except KeyError as error:
        raise LicenseRefused(
            catalog_id,
            f"the {AUTHORITY_DISTRIBUTION} authority holds no licence record "
            f"for {catalog_id}") from error

    try:
        root = receipt_store_root()
    except Exception as error:
        raise LicenseRefused(
            catalog_id,
            f"the licence receipt store could not be located "
            f"({_detail(error)})") from error
    if not os.path.isdir(root):
        raise LicenseRefused(
            catalog_id,
            f"no licence receipt for {catalog_id}: there is no receipt store "
            f"at {root}")
    store = _opened_store(kilix_license, root)

    if manifest_digest is not None:
        try:
            return kilix_license.require(
                kilix_license.AssetRef(
                    id=catalog_id,
                    record_digest=record.digest,
                    manifest_digest=manifest_digest),
                records=records, store=store)
        except Exception as error:
            raise LicenseRefused(
                catalog_id,
                f"no licence receipt in {root} covers {catalog_id} "
                f"({_detail(error)})") from error

    try:
        candidates = store.for_record(record.digest)
    except Exception as error:
        raise LicenseRefused(
            catalog_id,
            f"the licence receipts in {root} could not be read "
            f"({_detail(error)})") from error
    if not candidates:
        raise LicenseRefused(
            catalog_id, f"no licence receipt for {catalog_id} in {root}")

    refusal: Exception | None = None
    for receipt in candidates:
        try:
            if kilix_license.covers(
                    record, receipt, manifest_digest=receipt.manifest_digest):
                return receipt
        except Exception as error:
            refusal = error
    raise LicenseRefused(
        catalog_id,
        f"the licence receipt for {catalog_id} in {root} does not cover this "
        f"licence record ({_detail(refusal)})")


def _detail(error: Exception | None) -> str:
    """Return one short line naming what the authority refused."""
    if error is None:
        return "no receipt covered it"
    text = str(error).strip().splitlines()
    name = type(error).__name__
    return f"{name}: {text[0]}" if text and text[0] else name
