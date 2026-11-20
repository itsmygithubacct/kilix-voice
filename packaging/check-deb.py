#!/usr/bin/env python3
"""check-deb.py DEB : the kilix-voice archive package is the read-aloud runtime."""

import subprocess
import sys
import tempfile
from pathlib import Path


def fail(message: str) -> None:
    print(f"check-deb: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    if len(sys.argv) != 2:
        fail("usage: check-deb.py DEB")
    deb = Path(sys.argv[1])
    fields = subprocess.run(
        ["dpkg-deb", "-f", str(deb), "Package", "Version", "Architecture", "Depends"],
        check=True, capture_output=True, text=True).stdout
    if "Package: kilix-voice\n" not in fields:
        fail(f"unexpected package fields:\n{fields}")
    if "Version: 0.1.6+packaging1\n" not in fields:
        fail(f"unexpected version:\n{fields}")
    if "Architecture: all\n" not in fields:
        fail(f"unexpected architecture:\n{fields}")
    if "espeak-ng" not in fields:
        fail(f"read-aloud dependency missing:\n{fields}")
    with tempfile.TemporaryDirectory(prefix="kilix-voice-deb.") as raw:
        root = Path(raw)
        subprocess.run(["dpkg-deb", "-x", str(deb), str(root)], check=True)
        lib = root / "usr/lib/kilix-voice"
        if (lib / "VERSION").read_text() != "0.1.6\n":
            fail("library version differs")
        if not (lib / "voicelib/__init__.py").is_file():
            fail("library missing")
        if (lib / "tests").exists() or (root / "usr/share/kilix-voice/tests").exists():
            fail("tests were packaged")
        banned = list(root.rglob("*.onnx")) + list(root.rglob("*.gguf"))
        if banned:
            fail(f"weights were packaged: {banned[0]}")
        for name in ("kilix-tts", "kilix-stt", "kilix-voiced"):
            command = root / "usr/bin" / name
            if not command.is_file() or command.is_symlink():
                fail(f"{name} is not an installed command")
            shown = subprocess.run(
                [str(command), "--version"],
                check=True, capture_output=True, text=True).stdout.strip()
            if shown != f"{name} 0.1.6":
                fail(f"{name} version output is {shown!r}")
    print("check-deb: passed")


if __name__ == "__main__":
    main()
