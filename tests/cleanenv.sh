#!/bin/sh
# Run a command with no inherited environment: a fresh temp HOME and XDG tree,
# PATH=/usr/bin:/bin, and no KILIX_* / GPU_TERMINAL_* variable at all.
#
# The suite must give the same result here as on a machine with voice
# providers installed. An independent review found a test that passed only
# where kilix-piper-tts was on PATH, and a mutation battery that counted every
# mutant as killed wherever it was not. This is that review's runner, changed
# to wait for the command, remove its temporary tree, and exit with the
# command's own status rather than exec'ing and leaving the tree behind.
#
#     make test-clean
#     tests/cleanenv.sh /usr/bin/python3 -m unittest discover -s tests -t .
T=$(mktemp -d) || exit 1
mkdir -p "$T/home" "$T/data" "$T/config" "$T/runtime" "$T/state" "$T/cache"
chmod 700 "$T/runtime"
env -i PATH=/usr/bin:/bin LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 \
  HOME="$T/home" XDG_DATA_HOME="$T/data" XDG_CONFIG_HOME="$T/config" \
  XDG_RUNTIME_DIR="$T/runtime" XDG_STATE_HOME="$T/state" XDG_CACHE_HOME="$T/cache" \
  "$@"
status=$?
rm -rf "$T"
exit "$status"
