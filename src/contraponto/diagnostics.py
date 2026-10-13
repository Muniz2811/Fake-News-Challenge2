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


def install_correlation_record_factory() -> None:
    """Acrescenta correlation_id a todo LogRecord, inclusive bibliotecas."""
    factory = logging.getLogRecordFactory()
    if getattr(factory, "_contraponto_correlation", False):
        return

    def record_factory(*args, **kwargs):
        record = factory(*args, **kwargs)
        record.correlation_id = _correlation_id.get()
        return record

    record_factory._contraponto_correlation = True
    logging.setLogRecordFactory(record_factory)


@contextmanager
def investigation_context(correlation_id: str | None = None):
    """Cria contexto UUIDv4 ou herda o ID da investigação corrente."""
    install_correlation_record_factory()
    atual = _correlation_id.get()
    if correlation_id is not None:
        identificador = uuid.UUID(correlation_id)
        if identificador.version != 4:
            raise ValueError("O identificador técnico deve ser um UUIDv4.")
        valor = str(identificador)
    else:
        valor = atual if atual != "-" else str(uuid.uuid4())
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
def log_stage(logger: logging.Logger, stage: str, *, error_category: str | None = None):
    """Emite início/fim e duração sem registrar payloads nem exceções textuais."""
    install_correlation_record_factory()
    started = time.perf_counter()
    logger.info("stage=%s event=started", stage)
    try:
        yield
    except Exception as exc:
        elapsed = time.perf_counter() - started
        logger.error("stage=%s event=failed duration_seconds=%.6f duration_ms=%.3f error_type=%s error_category=%s",
                     stage, elapsed, elapsed * 1000, type(exc).__name__, error_category or _error_category(exc))
        raise
    else:
        elapsed = time.perf_counter() - started
        logger.info("stage=%s event=completed duration_seconds=%.6f duration_ms=%.3f",
                    stage, elapsed, elapsed * 1000)
