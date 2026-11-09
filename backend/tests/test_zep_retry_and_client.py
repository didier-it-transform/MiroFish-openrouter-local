from pathlib import Path
from types import SimpleNamespace

import pytest
from zep_cloud.core.api_error import ApiError as ZepApiError

from app.utils import zep


def test_permanent_zep_errors_fail_without_retry():
    calls = []

    def operation():
        calls.append(True)
        raise ZepApiError(status_code=400, body={"message": "bad query"})

    with pytest.raises(ZepApiError):
        zep.call_zep_read_with_retry(
            operation,
            operation_name="permanent failure",
            sleep=lambda _seconds: None,
        )

    assert len(calls) == 1


def test_rate_limit_retry_respects_retry_after():
    calls = []
    sleeps = []

    def operation():
        calls.append(True)
        if len(calls) == 1:
            raise ZepApiError(
                status_code=429,
                headers={"Retry-After": "7"},
                body={"message": "slow down"},
            )
        return "ok"

    result = zep.call_zep_read_with_retry(
        operation,
        operation_name="rate limited read",
        sleep=sleeps.append,
    )

    assert result == "ok"
    assert len(calls) == 2
    assert sleeps == [7.0]


def test_zep_client_is_shared_across_calls(monkeypatch):
    created = []

    def fake_zep(**kwargs):
        created.append(kwargs)
        return SimpleNamespace(kwargs=kwargs)

    monkeypatch.setattr(zep, "Zep", fake_zep)
    zep.clear_zep_client_cache()

    first = zep.get_zep_client("test-key")
    second = zep.get_zep_client("test-key")

    assert first is second
    assert len(created) == 1
    zep.clear_zep_client_cache()


def test_zep_timeout_policy_is_not_exposed_in_env_example():
    env_example = Path(__file__).resolve().parents[2] / ".env.example"
    contents = env_example.read_text(encoding="utf-8")

    assert "ZEP_REQUEST_TIMEOUT_SECONDS" not in contents
    assert "ZEP_INGESTION_TIMEOUT_SECONDS" not in contents
