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

Importing this module performs no filesystem, subprocess, network or authority
work. Everything happens inside the call.
"""

from __future__ import annotations

import os

from . import paths

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
# one component. The environment override exists so a deployment can put the
# store somewhere else; it cannot make an absent receipt cover anything.
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


def content_asset_id(catalog_id: str) -> str:
    """Return the id kilix-content files this model's weights under."""
    return CONTENT_ASSET_ID.get(catalog_id, catalog_id)


def accept_command(catalog_id: str) -> str:
    """Return the first-use command that shows the licence and records consent."""
    return f"{ACCEPT_COMMAND} {content_asset_id(catalog_id)}"


class LicenseRefused(RuntimeError):
    """Weights may not be fetched: no receipt covers this model."""

    def __init__(self, catalog_id: str, reason: str) -> None:
        self.catalog_id = catalog_id
        self.reason = reason
        super().__init__(
            f"refusing to fetch the {catalog_id} model weights: {reason}. "
            "Model weights are fetched only after their licence has been "
            f"shown and accepted. Run: {accept_command(catalog_id)}")


class AuthorityUnavailable(LicenseRefused):
    """The licence authority is not installed, so nothing can be verified."""


def receipt_store_root() -> str:
    """Return the directory the licence authority publishes receipts into."""
    override = os.environ.get(ENV_RECEIPTS)
    if override:
        return os.path.abspath(os.path.expanduser(override))
    return os.path.join(paths.gpu_terminal_home(), RECEIPTS_LEAF)


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
    missing = [name for name in AUTHORITY_API if not hasattr(kilix_license, name)]
    if missing:
        raise AuthorityUnavailable(
            catalog_id,
            f"the importable {AUTHORITY_MODULE} does not expose the licence "
            f"authority's API (missing {', '.join(sorted(missing))})")
    return kilix_license


def require_covering_receipt(catalog_id: str, *, manifest_digest: str | None = None):
    """Return the receipt that covers ``catalog_id``, or raise LicenseRefused.

    Nothing is written anywhere on the refusing path: the receipt store is read
    only if it already exists, and no directory under the model store is
    created, opened for writing, or touched at all.

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
    records = kilix_license.RecordIndex(kilix_license.load_determined_records())
    try:
        record = records.by_id(catalog_id)
    except KeyError as error:
        raise LicenseRefused(
            catalog_id,
            f"the {AUTHORITY_DISTRIBUTION} authority holds no licence record "
            f"for {catalog_id}") from error

    root = receipt_store_root()
    if not os.path.isdir(root):
        raise LicenseRefused(
            catalog_id,
            f"no licence receipt for {catalog_id}: there is no receipt store "
            f"at {root}")
    store = kilix_license.ReceiptStore(root)

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
