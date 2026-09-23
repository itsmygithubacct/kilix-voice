.POSIX:
PYTHON ?= python3
# The scrubbed-environment run uses the pinned system interpreter, the one the
# independent reviews ran; PATH inside tests/cleanenv.sh is /usr/bin:/bin.
CLEAN_PYTHON ?= /usr/bin/python3
PREFIX ?= $(HOME)/.local
# S04: the accelerator lease has one home, the kilix-device-lease component of
# kilix-system-monitor. kilix-voice imports it and carries no copy, so its
# tests need that source on the import path; tests/test_accelerator.py fails,
# naming this variable, when it is not.
LEASE_SRC ?= ../../kilix-system-monitor/components/kilix-device-lease/src
# V-ACC: the licence authority has one home too, kilix-license. kilix-voice
# asks it whether a receipt covers a model before any weight fetch, and copies
# none of its records, so its tests need that source on the import path;
# tests/test_weight_licence.py fails, naming this variable, when it is not.
LICENSE_SRC ?= ../../kilix-modules/kilix-license/src
CONTENT_SRC ?= ../../kilix-modules/kilix-content/src

SUITE_PYTHONPATH = $(abspath $(LEASE_SRC)):$(abspath $(LICENSE_SRC)):$(abspath $(CONTENT_SRC))

.PHONY: all test test-clean lint install uninstall clean

all: test

test:
	PYTHONPATH="$(SUITE_PYTHONPATH)" $(PYTHON) -m unittest discover -s tests -t . -v

# Same suite, no inherited environment: no KILIX_* or GPU_TERMINAL_* variable,
# a temporary HOME and XDG tree. Both targets must report no failures.
test-clean:
	./tests/cleanenv.sh /usr/bin/env PYTHONPATH="$(SUITE_PYTHONPATH)" \
	  $(CLEAN_PYTHON) -m unittest discover -s tests -t .

lint:
	$(PYTHON) -m compileall -q voicelib tests
	$(PYTHON) -c "import ast,sys,pathlib; [ast.parse(p.read_text()) for p in pathlib.Path('.').glob('kilix-*')]"

install:
	install -d $(PREFIX)/bin $(PREFIX)/lib/kilix-voice/voicelib
	install -m 0755 kilix-tts kilix-stt kilix-voiced $(PREFIX)/bin/
	install -m 0644 VERSION $(PREFIX)/lib/kilix-voice/VERSION
	install -m 0644 voicelib/*.py $(PREFIX)/lib/kilix-voice/voicelib/

# Remove exactly what install copied: the three commands, VERSION and the
# voicelib modules this checkout has, plus those modules' bytecode. Nothing is
# removed unless every present target is a regular file with this checkout's
# bytes. It refuses when an install directory is this checkout or lies inside
# it, or a target is the source file itself, so it cannot delete the files it
# compares against. install never creates symlinks, so a symlink at a target
# (kilix's managed ~/.local/bin entrypoints are symlinks into its store) is
# not ours and is left in place.
uninstall:
	@set -eu; \
	here=$$(pwd -P); \
	lib="$(PREFIX)/lib/kilix-voice"; \
	target_for() { \
		case "$$1" in \
			kilix-*) target="$(PREFIX)/bin/$$1" ;; \
			*) target="$$lib/$$1" ;; \
		esac; \
	}; \
	for dir in "$(PREFIX)/bin" "$$lib" "$$lib/voicelib" \
			"$$lib/voicelib/__pycache__"; do \
		[ -d "$$dir" ] || continue; \
		resolved=$$(CDPATH= cd -P -- "$$dir" && pwd -P); \
		case "$$resolved/" in \
			"$$here"/*) \
				echo "refusing to uninstall: $$dir is this checkout or lies inside it ($$resolved)" >&2; \
				exit 1 ;; \
		esac; \
	done; \
	for source in kilix-tts kilix-stt kilix-voiced VERSION voicelib/*.py; do \
		target_for "$$source"; \
		if [ -L "$$target" ]; then \
			echo "leaving $$target: a symlink, which make install never creates" >&2; \
		elif [ -e "$$target" ] && [ "$$source" -ef "$$target" ]; then \
			echo "refusing to uninstall: $$target is this checkout's own $$source" >&2; \
			exit 1; \
		elif [ -e "$$target" ] && { [ ! -f "$$target" ] || \
				! cmp -s "$$source" "$$target"; }; then \
			echo "refusing to remove modified or foreign file: $$target" >&2; \
			exit 1; \
		fi; \
	done; \
	for source in kilix-tts kilix-stt kilix-voiced VERSION voicelib/*.py; do \
		target_for "$$source"; \
		if [ -f "$$target" ] && [ ! -L "$$target" ]; then \
			rm -f "$$target"; \
		fi; \
	done; \
	if [ ! -L "$$lib/voicelib/__pycache__" ]; then \
		for source in voicelib/*.py; do \
			stem=$${source#voicelib/}; \
			stem=$${stem%.py}; \
			rm -f "$$lib/voicelib/__pycache__/$$stem".*.pyc; \
		done; \
	fi; \
	rmdir "$$lib/voicelib/__pycache__" "$$lib/voicelib" "$$lib" \
		2>/dev/null || true

clean:
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
