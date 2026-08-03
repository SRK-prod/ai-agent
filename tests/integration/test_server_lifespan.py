"""The backend must keep serving /health, /metrics, /enroll even when
MeetingPipeline can't start (e.g. a missing credential like HF_TOKEN) --
discovered by actually running the backend without credentials configured.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

import meeting_copilot.server.api as api_module


def test_health_ok_when_pipeline_fails_to_start():
    with patch.object(
        api_module, "MeetingPipeline", side_effect=RuntimeError("HF_TOKEN missing")
    ), TestClient(api_module.app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "pipeline": "not_started"}


def test_health_reports_running_when_pipeline_starts():
    fake_pipeline = MagicMock()
    fake_pipeline.stop = AsyncMock()
    with patch.object(api_module, "MeetingPipeline", return_value=fake_pipeline):
        with TestClient(api_module.app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json() == {"status": "ok", "pipeline": "running"}
        fake_pipeline.start.assert_called_once()
