.PHONY: setup venv services services-down enroll ingest run start stop status run-backend run-ui test test-fast lint typecheck format clean download-models package

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

venv:
	python3.13 -m venv $(VENV)
	$(PIP) install --upgrade pip

setup: venv
	$(PIP) install -e ".[dev]"
	$(PY) -m playwright install chromium

services:
	docker compose up -d qdrant redis

services-down:
	docker compose down

download-models:
	$(PY) scripts/download_models.py

enroll:
	$(PY) scripts/enroll_voice.py

ingest:
	$(PY) scripts/ingest_knowledge.py --topics configs/topics.yaml

qa-bank:
	$(PY) scripts/build_qa_bank.py

run-backend:
	$(VENV)/bin/uvicorn meeting_copilot.server.main:app --loop uvloop --host $${MEETING_COPILOT_HOST:-127.0.0.1} --port $${MEETING_COPILOT_PORT:-8765}

run-ui:
	$(PY) -m meeting_copilot.desktop.app

run: services
	scripts/start.sh

# Detached start: backend + overlay in the background, terminal stays free.
start: services
	@pkill -f "uvicorn meeting_copilot.server.main" 2>/dev/null; pkill -f "meeting_copilot.desktop.app" 2>/dev/null; sleep 1; true
	nohup $(VENV)/bin/uvicorn meeting_copilot.server.main:app --loop uvloop --host $${MEETING_COPILOT_HOST:-127.0.0.1} --port $${MEETING_COPILOT_PORT:-8765} > logs/backend.log 2>&1 &
	@echo "waiting for backend..." && for i in $$(seq 1 30); do curl -s -m 2 localhost:$${MEETING_COPILOT_PORT:-8765}/health 2>/dev/null | grep -q running && break; sleep 1; done
	nohup $(PY) -m meeting_copilot.desktop.app > logs/overlay.log 2>&1 &
	@sleep 2 && $(MAKE) --no-print-directory status

# Stop backend + overlay (leaves Qdrant/Redis containers running; use services-down for those).
stop:
	-pkill -f "uvicorn meeting_copilot.server.main"
	-pkill -f "meeting_copilot.desktop.app"
	@echo "stopped (docker services still up; 'make services-down' to stop them too)"

status:
	@curl -s -m 2 localhost:$${MEETING_COPILOT_PORT:-8765}/health 2>/dev/null || echo "backend: not running"
	@echo
	@pgrep -fl "meeting_copilot.desktop.app" >/dev/null 2>&1 && echo "overlay: running" || echo "overlay: not running"

test-fast:
	$(PY) -m pytest -m "not slow and not e2e"

test:
	$(PY) -m pytest

lint:
	$(VENV)/bin/ruff check src tests

format:
	$(VENV)/bin/ruff format src tests

typecheck:
	$(VENV)/bin/mypy src

package:
	$(VENV)/bin/pyinstaller meeting-copilot.spec --noconfirm

clean:
	rm -rf $(VENV) build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
