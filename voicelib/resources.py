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


def fits(profile: dict, *, available_vram_mib: int | None = None,
         available_ram_mib: int | None = None) -> bool:
    """Return whether measured demand fits the stated headroom.

    Absent headroom is UNKNOWN, not infinite: a caller that cannot measure its
    own device gets False rather than an optimistic True.
    """
    checked = validate(profile)
    measured = checked["measured"]
    if checked["device_class"] != DEVICE_CPU:
        if available_vram_mib is None:
            return False
        if measured["peak_vram_mib"] > available_vram_mib:
            return False
    if "peak_ram_mib" in measured:
        if available_ram_mib is None:
            return False
        if measured["peak_ram_mib"] > available_ram_mib:
            return False
    return True
