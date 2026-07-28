.POSIX:
PYTHON ?= python3
PREFIX ?= $(HOME)/.local

.PHONY: all test lint install clean

all: test

test:
	$(PYTHON) -m unittest discover -s tests -t . -v

lint:
	$(PYTHON) -m compileall -q voicelib tests
	$(PYTHON) -c "import ast,sys,pathlib; [ast.parse(p.read_text()) for p in pathlib.Path('.').glob('kilix-*')]"

install:
	install -d $(PREFIX)/bin
	install -m 0755 kilix-tts kilix-stt kilix-voiced $(PREFIX)/bin/

clean:
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
