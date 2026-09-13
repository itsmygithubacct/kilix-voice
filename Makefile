.POSIX:
PYTHON ?= python3
# The scrubbed-environment run uses the pinned system interpreter, the one the
# independent reviews ran; PATH inside tests/cleanenv.sh is /usr/bin:/bin.
CLEAN_PYTHON ?= /usr/bin/python3
PREFIX ?= $(HOME)/.local

.PHONY: all test test-clean lint install clean

all: test

test:
	$(PYTHON) -m unittest discover -s tests -t . -v

# Same suite, no inherited environment: no KILIX_* or GPU_TERMINAL_* variable,
# a temporary HOME and XDG tree. Both targets must report no failures.
test-clean:
	./tests/cleanenv.sh $(CLEAN_PYTHON) -m unittest discover -s tests -t .

lint:
	$(PYTHON) -m compileall -q voicelib tests
	$(PYTHON) -c "import ast,sys,pathlib; [ast.parse(p.read_text()) for p in pathlib.Path('.').glob('kilix-*')]"

install:
	install -d $(PREFIX)/bin $(PREFIX)/lib/kilix-voice/voicelib
	install -m 0755 kilix-tts kilix-stt kilix-voiced $(PREFIX)/bin/
	install -m 0644 VERSION $(PREFIX)/lib/kilix-voice/VERSION
	install -m 0644 voicelib/*.py $(PREFIX)/lib/kilix-voice/voicelib/

clean:
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
