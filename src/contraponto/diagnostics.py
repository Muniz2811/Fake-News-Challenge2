"""Logging técnico com correlação por investigação e etapas cronometradas.

Este módulo deliberadamente não aceita dados de negócio como argumentos de log.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import logging
import time
import uuid

_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="-")
_factory_installed = False


def install_correlation_record_factory() -> None:
    """Acrescenta correlation_id a todo LogRecord, inclusive bibliotecas."""
    global _factory_installed
    if _factory_installed:
        return
    factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = factory(*args, **kwargs)
        record.correlation_id = _correlation_id.get()
        return record

    logging.setLogRecordFactory(record_factory)
    _factory_installed = True


@contextmanager
def investigation_context(correlation_id: str | None = None):
    """Cria contexto UUIDv4 ou herda o ID da investigação corrente."""
    install_correlation_record_factory()
    atual = _correlation_id.get()
    valor = correlation_id or (atual if atual != "-" else str(uuid.uuid4()))
    token = _correlation_id.set(valor)
    try:
        yield valor
    finally:
        _correlation_id.reset(token)


def current_correlation_id() -> str:
    return _correlation_id.get()


def logged_async_stage(stage: str):
    """Decorador para criar/propagar investigação em uma operação async."""
    def decorate(function):
        @wraps(function)
        async def wrapped(*args, **kwargs):
            with investigation_context():
                with log_stage(logging.getLogger(function.__module__), stage):
                    return await function(*args, **kwargs)
        return wrapped
    return decorate


def _error_category(exc: BaseException) -> str:
    nome = type(exc).__name__.casefold()
    if isinstance(exc, (KeyError, ValueError)) and any(x in str(exc).casefold() for x in ("api key", "chave", "token", "credential")):
        return "CONFIGURATION_FAILURE"
    if any(x in nome for x in ("timeout", "connection", "apierror", "serviceunavailable", "ratelimit")):
        return "EXTERNAL_SERVICE_UNAVAILABLE"
    return "INTERNAL_ERROR"


@contextmanager
def log_stage(logger: logging.Logger, stage: str):
    """Emite início/fim e duração sem registrar payloads nem exceções textuais."""
    install_correlation_record_factory()
    started = time.perf_counter()
    logger.info("stage=%s event=started", stage)
    try:
        yield
    except Exception as exc:
        elapsed = time.perf_counter() - started
        logger.error("stage=%s event=failed duration_seconds=%.6f duration_ms=%.3f error_type=%s error_category=%s",
                     stage, elapsed, elapsed * 1000, type(exc).__name__, _error_category(exc))
        raise
    else:
        elapsed = time.perf_counter() - started
        logger.info("stage=%s event=completed duration_seconds=%.6f duration_ms=%.3f",
                    stage, elapsed, elapsed * 1000)
