"""The speech resource profile: what a model needs, measured, not guessed.

C06 requires a device class, C12 a measured resource profile, C18 that the
profile conforms to a frozen F106 shape. None existed. Note carefully what this
module is NOT: the artefact currently bound to F104's F106-RESOURCE-PROFILE
return is `kilix.trusted-launcher.profile/v1`, a launcher command/replay profile
with no resource field of any kind. This declares the *speech* resource shape
those three requirements actually describe. Binding it is a separate,
joint act -- see the F104 P1 execution assessment.

Every figure here is MEASURED on a named host, never estimated. A guessed VRAM
number that is too low turns into an OOM at the worst moment, and one that is
too high silently disqualifies hardware that would have worked; neither failure
announces itself as a bad guess.

Importing this module performs no filesystem, subprocess or device work.
"""

from __future__ import annotations

import re

RESOURCE_SCHEMA = "kilix.speech.resource-profile/v1"

DEVICE_CPU = "cpu"
DEVICE_CUDA = "cuda"
DEVICE_VULKAN = "vulkan"
DEVICE_CLASSES = (DEVICE_CPU, DEVICE_CUDA, DEVICE_VULKAN)

# Measured, so each carries the host it was measured on and the date.
_REQUIRED = ("schema", "device_class", "measured")
_MEASURED_INTS = ("peak_vram_mib", "peak_ram_mib", "model_bytes")
_HOST = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class ResourceError(ValueError):
    """A malformed or unmeasured resource profile."""


def validate(profile: object) -> dict:
    """Return the profile if it conforms to the frozen shape, else raise.

    C18. Unknown keys are preserved, as in the catalog reader: a newer producer
    must not break an older consumer. What is refused is a missing or unmeasured
    figure, because that is the case where a consumer would act on nothing.
    """
    if not isinstance(profile, dict):
        raise ResourceError(
            f"a resource profile must be an object, got "
            f"{type(profile).__name__}.")
    for key in _REQUIRED:
        if key not in profile:
            raise ResourceError(f"resource profile is missing {key!r}.")
    if profile["schema"] != RESOURCE_SCHEMA:
        raise ResourceError(
            f"resource profile schema is {profile['schema']!r}; this reader "
            f"speaks {RESOURCE_SCHEMA!r}.")
    device = profile["device_class"]
    if device not in DEVICE_CLASSES:
        raise ResourceError(
            f"device_class must be one of: {', '.join(DEVICE_CLASSES)}, got "
            f"{device!r}.")
    measured = profile["measured"]
    if not isinstance(measured, dict):
        raise ResourceError(
            f"'measured' must be an object, got {type(measured).__name__}.")
    host = measured.get("host")
    if not isinstance(host, str) or not _HOST.fullmatch(host):
        raise ResourceError(
            "'measured.host' must name the host the figures were taken on, "
            f"1-64 characters from [A-Za-z0-9._-]; got {host!r}. A figure "
            "without a host cannot be reproduced or challenged.")
    date = measured.get("date")
    if not isinstance(date, str) or not _DATE.fullmatch(date):
        raise ResourceError(
            f"'measured.date' must be YYYY-MM-DD, got {date!r}.")
    present = [k for k in _MEASURED_INTS if k in measured]
    if not present:
        raise ResourceError(
            "'measured' carries no figure at all; at least one of "
            f"{', '.join(_MEASURED_INTS)} is required. An empty profile would "
            "certify a model as having been measured when it has not.")
    for key in present:
        value = measured[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ResourceError(
                f"'measured.{key}' must be a non-negative integer, got "
                f"{value!r}.")
    if device == DEVICE_CPU and measured.get("peak_vram_mib"):
        raise ResourceError(
            "a cpu profile reports peak_vram_mib "
            f"{measured['peak_vram_mib']}; VRAM on a cpu device class is a "
            "measurement error, not a small number.")
    if device != DEVICE_CPU and "peak_vram_mib" not in measured:
        raise ResourceError(
            f"a {device} profile must report peak_vram_mib; that figure is the "
            "whole reason an accelerator profile exists.")
    return dict(profile)


# Every measured demand maps to the headroom that must cover it. Adding a
# measured figure without adding its headroom here makes fits() return False
# rather than silently ignoring the new demand -- unknown, not satisfied.
_DEMAND_TO_HEADROOM = {
    "peak_vram_mib": "available_vram_mib",
    "peak_ram_mib": "available_ram_mib",
    "model_bytes": "available_disk_bytes",
}


def _headroom(name: str, value: object) -> int | None:
    """Return a validated headroom figure, or None when not supplied."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        # floats included on purpose: NaN and inf are not headroom, and NaN in
        # particular makes every comparison false, which reads as "fits".
        raise ResourceError(
            f"{name} must be a non-negative integer or None, got {value!r}.")
    if value < 0:
        raise ResourceError(f"{name} must not be negative, got {value}.")
    return value


def fits(profile: dict, *, available_vram_mib: int | None = None,
         available_ram_mib: int | None = None,
         available_disk_bytes: int | None = None) -> bool:
    """Return whether EVERY measured demand is covered by supplied headroom.

    Absent headroom is UNKNOWN, not infinite. If the profile measured a demand
    and the caller did not supply the matching headroom, the answer is False --
    a caller that cannot measure its own device gets a refusal, not an
    optimistic yes.

    An earlier revision checked only the demands the profile happened to carry
    against only the headroom the caller happened to pass, so a profile
    measuring just model_bytes returned True with nothing supplied at all. That
    contradicted this docstring; an independent review caught it.
    """
    checked = validate(profile)
    supplied = {
        "available_vram_mib": _headroom("available_vram_mib", available_vram_mib),
        "available_ram_mib": _headroom("available_ram_mib", available_ram_mib),
        "available_disk_bytes": _headroom("available_disk_bytes", available_disk_bytes),
    }
    measured = checked["measured"]
    demands = {k: v for k, v in measured.items() if k in _DEMAND_TO_HEADROOM}
    if not demands:                      # validate() forbids this, belt and braces
        return False
    for demand, value in demands.items():
        name = _DEMAND_TO_HEADROOM[demand]
        headroom = supplied[name]
        if headroom is None:             # unknown, never infinite
            return False
        if value > headroom:
            return False
    return True
