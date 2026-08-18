.POSIX:
PYTHON ?= python3
PREFIX ?= $(HOME)/.local

.PHONY: all test lint install uninstall clean

all: test

test:
	$(PYTHON) -m unittest discover -s tests -t . -v

lint:
	$(PYTHON) -m compileall -q voicelib tests
	$(PYTHON) -c "import ast,sys,pathlib; [ast.parse(p.read_text()) for p in pathlib.Path('.').glob('kilix-*')]"

install:
	install -d $(PREFIX)/bin $(PREFIX)/lib/kilix-voice/voicelib
	install -m 0755 kilix-tts kilix-stt kilix-voiced $(PREFIX)/bin/
	install -m 0644 VERSION $(PREFIX)/lib/kilix-voice/VERSION
	install -m 0644 voicelib/*.py $(PREFIX)/lib/kilix-voice/voicelib/

uninstall:
	@set -eu; \
	for source in kilix-tts kilix-stt kilix-voiced VERSION voicelib/*.py; do \
		case "$$source" in \
			kilix-*) target="$(PREFIX)/bin/$$source" ;; \
			VERSION) target="$(PREFIX)/lib/kilix-voice/VERSION" ;; \
			*) target="$(PREFIX)/lib/kilix-voice/$$source" ;; \
		esac; \
		if [ -e "$$target" ] && ! cmp -s "$$source" "$$target"; then \
			echo "refusing to remove modified or foreign file: $$target" >&2; \
			exit 1; \
		fi; \
	done; \
	rm -f "$(PREFIX)/bin/kilix-tts" "$(PREFIX)/bin/kilix-stt" \
		"$(PREFIX)/bin/kilix-voiced" \
		"$(PREFIX)/lib/kilix-voice/VERSION"; \
	for source in voicelib/*.py; do \
		rm -f "$(PREFIX)/lib/kilix-voice/$$source"; \
	done; \
	rm -rf "$(PREFIX)/lib/kilix-voice/voicelib/__pycache__"; \
	rmdir "$(PREFIX)/lib/kilix-voice/voicelib" \
		"$(PREFIX)/lib/kilix-voice" 2>/dev/null || true

clean:
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
