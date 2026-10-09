"""Testes de correlação, duração e privacidade da instrumentação."""
import logging
import os
import sys
import uuid

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "contraponto"))
from diagnostics import investigation_context, log_stage


def test_stage_logs_correlation_and_elapsed_time(caplog):
    logger = logging.getLogger("diagnostics_test")
    caplog.set_level(logging.INFO, logger=logger.name)
    with investigation_context() as correlation_id:
        with log_stage(logger, "unit_stage"):
            pass

    uuid.UUID(correlation_id, version=4)
    records = [record for record in caplog.records if record.name == logger.name]
    assert [record.getMessage().split("event=")[1].split()[0] for record in records] == ["started", "completed"]
    assert all(record.correlation_id == correlation_id for record in records)
    assert "duration_seconds=" in records[-1].getMessage()
    assert "duration_ms=" in records[-1].getMessage()


def test_stage_failure_is_categorized_without_exception_payload(caplog):
    logger = logging.getLogger("diagnostics_failure_test")
    caplog.set_level(logging.ERROR, logger=logger.name)
    sensitive = "mensagem privada e identificador pessoal"
    with investigation_context():
        with pytest.raises(RuntimeError):
            with log_stage(logger, "failing_stage"):
                raise RuntimeError(sensitive)

    output = " ".join(record.getMessage() for record in caplog.records)
    assert "error_category=INTERNAL_ERROR" in output
    assert "RuntimeError" in output
    assert sensitive not in output


def test_external_service_failures_have_distinct_category(caplog):
    logger = logging.getLogger("diagnostics_external_test")
    caplog.set_level(logging.ERROR, logger=logger.name)
    with investigation_context():
        with pytest.raises(TimeoutError):
            with log_stage(logger, "remote_call"):
                raise TimeoutError("private payload")
    assert "error_category=EXTERNAL_SERVICE_UNAVAILABLE" in caplog.text
    assert "private payload" not in caplog.text
