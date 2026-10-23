"""Sala web de demonstração: HTTP, histórico em memória e o mesmo MotorIA."""

import asyncio
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import math
import os
from pathlib import Path
import secrets
import time
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from motor_ia import MotorIA, ResultadoAnalise
from verifier import ROTULOS_CATEGORIAS

logger = logging.getLogger(__name__)
ASSETS = Path(__file__).with_name("web_static")
BRAND_ASSETS = Path(__file__).resolve().parents[2] / "assets"
BRAND_FILES = {
    "Logomarca.png", "Logotipo.png", "Wallpaper.png",
    "CrimsonText-Regular.ttf", "CrimsonText-OFL.txt",
}
COOKIE = "contraponto_session"
PROFILES = {f"p{index}": f"Pessoa {index}" for index in range(1, 5)}


@dataclass(frozen=True)
class WebSettings:
    """Limites pequenos para uma única sala e um único processo."""

    access_code: str = ""
    room_name: str = "Grupo de demonstração"
    max_messages: int = 80
    queue_size: int = 8
    max_sessions: int = 4
    session_seconds: int = 12 * 60 * 60
    cooldown_seconds: float = 5

    @classmethod
    def from_env(cls):
        return cls(
            access_code=os.getenv("WEB_ACCESS_CODE", ""),
            room_name=os.getenv("WEB_ROOM_NAME", "").strip()[:80] or cls.room_name,
        )


class JoinRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    profile_id: Literal["p1", "p2", "p3", "p4"]
    access_code: str = Field(default="", max_length=128, strict=True)
    group_consent: bool = Field(default=False, strict=True)


class ForwardRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chat_id: Literal["p1", "p2", "p3", "p4"]


class MessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    text: str = Field(min_length=1, max_length=4000, strict=True)
    chat_id: str = Field(default="group", min_length=1, max_length=20, strict=True)


@dataclass
class Session:
    id: str
    name: str
    expires: float
    seen: float
    last_message: float = -float("inf")


@dataclass
class Room:
    settings: WebSettings
    sessions: dict[str, Session] = field(default_factory=dict)
    messages: dict[str, list[dict]] = field(default_factory=dict)
    pending_users: set[str] = field(default_factory=set)
    version: int = 0

    def __post_init__(self):
        self.queue = asyncio.Queue(maxsize=self.settings.queue_size)

    def session(self, request: Request) -> Session:
        token = request.cookies.get(COOKIE, "")
        session = self.sessions.get(token)
        if session is None or session.expires <= time.monotonic():
            self.sessions.pop(token, None)
            raise HTTPException(401, "Entre na sala para continuar.")
        session.seen = time.monotonic()
        return session

    def conversation(self, user: Session, chat_id: str) -> str:
        """Resolve só grupo, bot próprio ou uma conversa que inclua o usuário."""
        if chat_id == "group":
            return "group"
        if chat_id == "bot":
            return f"bot:{user.id}"
        if chat_id in PROFILES and chat_id != user.id:
            return "dm:" + ":".join(sorted((user.id, chat_id)))
        raise HTTPException(404, "Conversa não encontrada.")

    def profiles(self):
        now = time.monotonic()
        online = {user.id for user in self.sessions.values() if user.expires > now and now - user.seen < 60}
        return [{"id": uid, "name": name, "online": uid in online} for uid, name in PROFILES.items()]

    def check_queue(self, user: Session):
        if user.id in self.pending_users or time.monotonic() - user.last_message < self.settings.cooldown_seconds:
            raise HTTPException(429, "Aguarde seu processamento atual antes de solicitar outro.")
        if self.queue.full():
            raise HTTPException(429, "A fila está cheia. Tente novamente em instantes.")

    def append_message(self, key: str, user: Session, text: str) -> dict:
        history = self.messages.setdefault(key, [])
        if len(history) >= self.settings.max_messages:
            old = next((item for item in history if item["status"] in
                        {"done", "error", "sent"}), None)
            if old is None:
                raise HTTPException(429, "Aguarde os processamentos em andamento.")
            history.remove(old)
        message = {
            "id": secrets.token_hex(12), "author_id": user.id, "author": user.name,
            "text": text, "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "sent", "analysis": None,
        }
        history.append(message)
        self.version += 1
        return message

    def enqueue(self, message: dict, user: Session):
        message["status"] = "queued"
        message.pop("error", None)
        self.pending_users.add(user.id)
        self.queue.put_nowait((message, user.id))
        user.last_message = time.monotonic()
        self.version += 1

    async def verify_messages(self, motor):
        """Serializa investigações para preservar memória e estado dos provedores."""
        while True:
            message, requester_id = await self.queue.get()
            message["status"] = "verifying"
            self.version += 1
            try:
                result = await motor.analisar_texto(message["text"])
                message["analysis"] = serialize_result(result)
                message["status"] = "done"
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("stage=web_analysis error_type=%s", type(exc).__name__)
                message["status"] = "error"
                message["error"] = "A análise ficou indisponível. Tente novamente."
            finally:
                self.pending_users.discard(requester_id)
                self.queue.task_done()
                self.version += 1


def serialize_result(result: ResultadoAnalise) -> dict:
    """Expõe o contrato do motor, preservando fontes por alegação e separação factual."""
    score = result.score_linguistico
    return {
        "language": {
            "text": result.analise_linguistica,
            "model": result.nome_modelo_linguistico,
            "score": score if type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1 else None,
        },
        "claims": [{
            "id": verdict.alegacao_id,
            "text": verdict.texto_alegacao,
            "category": verdict.categoria,
            "label": ROTULOS_CATEGORIAS.get(verdict.categoria, "não verificada"),
            "rationale": verdict.justificativa,
            "sources": verdict.fontes_citadas,
        } for verdict in result.vereditos],
        "summary": result.resultado_factual,
        "warnings": result.etapas_indisponiveis,
        "requires_review": result.requer_revisao,
    }


def create_app(*, motor=None, settings: WebSettings | None = None) -> FastAPI:
    """Permite injetar o motor nos testes sem executar modelos ou APIs reais."""
    settings = settings if settings is not None else WebSettings.from_env()

    @asynccontextmanager
    async def lifespan(app):
        room = Room(settings)
        app.state.room = room
        engine = motor if motor is not None else await asyncio.to_thread(MotorIA)
        worker = asyncio.create_task(room.verify_messages(engine), name="web-verification")
        try:
            yield
        finally:
            worker.cancel()
            with suppress(asyncio.CancelledError):
                await worker

    app = FastAPI(title="Contraponto — sala de demonstração", lifespan=lifespan, docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def response_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "font-src 'self'; img-src 'self' data:; "
            "connect-src 'self'; frame-ancestors 'none'; "
            "base-uri 'none'; form-action 'self'"
        )
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    async def index():
        return HTMLResponse((ASSETS / "index.html").read_text(encoding="utf-8"))

    @app.get("/brand/{filename}")
    async def brand_asset(filename: str):
        if filename not in BRAND_FILES or not (BRAND_ASSETS / filename).is_file():
            raise HTTPException(404, "Arquivo de identidade visual não encontrado.")
        return FileResponse(BRAND_ASSETS / filename, media_type="font/ttf" if filename.endswith(".ttf") else None)

    @app.get("/healthz")
    async def health():
        return {"status": "ok"}

    @app.get("/api/config")
    async def config(request: Request):
        return {"room_name": settings.room_name, "code_required": bool(settings.access_code),
                "profiles": request.app.state.room.profiles()}

    @app.post("/api/session", status_code=201)
    async def join(body: JoinRequest, request: Request, response: Response):
        if not body.group_consent:
            raise HTTPException(403, "Confirme que as mensagens do grupo podem ser analisadas por solicitação.")
        if settings.access_code and not secrets.compare_digest(
            body.access_code.encode("utf-8"), settings.access_code.encode("utf-8"),
        ):
            raise HTTPException(403, "Código de acesso incorreto.")
        room = request.app.state.room
        now = time.monotonic()
        room.sessions = {token: item for token, item in room.sessions.items() if item.expires > now}
        previous = request.cookies.get(COOKIE)
        occupied = any(user.id == body.profile_id and token != previous and now - user.seen < 60
                       for token, user in room.sessions.items())
        if occupied:
            raise HTTPException(409, "Este perfil já está em uso. Escolha outra pessoa.")
        room.sessions = {token: user for token, user in room.sessions.items()
                         if token != previous and user.id != body.profile_id}
        if len(room.sessions) >= room.settings.max_sessions:
            raise HTTPException(503, "A sala atingiu o limite de participantes. Tente mais tarde.")
        token = secrets.token_urlsafe(32)
        session = Session(body.profile_id, PROFILES[body.profile_id], now + settings.session_seconds, now)
        room.sessions[token] = session
        room.version += 1
        response.set_cookie(COOKIE, token, max_age=settings.session_seconds, httponly=True,
                            samesite="strict", secure=request.url.scheme == "https")
        return {"id": session.id, "name": session.name}

    @app.get("/api/session")
    async def current_session(request: Request):
        session = request.app.state.room.session(request)
        return {"id": session.id, "name": session.name}

    @app.delete("/api/session", status_code=204)
    async def leave(request: Request, response: Response):
        room = request.app.state.room
        room.sessions.pop(request.cookies.get(COOKIE, ""), None)
        room.version += 1
        response.delete_cookie(COOKIE)

    @app.get("/api/messages")
    async def messages(request: Request, chat_id: str = "group", version: int = -1):
        room = request.app.state.room
        user = room.session(request)
        key = room.conversation(user, chat_id)
        return {
            "version": room.version,
            "messages": room.messages.get(key, []) if version != room.version else None,
            "profiles": room.profiles(),
        }

    @app.get("/api/chats")
    async def chats(request: Request):
        room = request.app.state.room
        user = room.session(request)
        return [{"id": "group", "name": settings.room_name, "type": "group"},
                {"id": "bot", "name": "Contraponto", "type": "bot"}] + [
                    {"id": uid, "name": name, "type": "private"} for uid, name in PROFILES.items() if uid != user.id
                ]

    @app.post("/api/messages", status_code=202)
    async def send(body: MessageRequest, request: Request):
        room = request.app.state.room
        user = room.session(request)
        key = room.conversation(user, body.chat_id)
        automatic = body.chat_id == "bot"
        if automatic:
            room.check_queue(user)
        message = room.append_message(key, user, body.text)
        if automatic:
            room.enqueue(message, user)
        return {"id": message["id"]}

    @app.post("/api/messages/{message_id}/verify", status_code=202)
    async def verify(message_id: str, request: Request):
        room = request.app.state.room
        user = room.session(request)
        message = next((item for item in room.messages.get("group", []) if item["id"] == message_id), None)
        if message is None:
            raise HTTPException(404, "Mensagem do grupo não encontrada.")
        if message["status"] not in {"sent", "error"}:
            raise HTTPException(409, "Esta mensagem já está em análise ou tem um resultado disponível.")
        room.check_queue(user)
        room.enqueue(message, user)
        return {"id": message["id"]}

    @app.post("/api/messages/{message_id}/forward", status_code=202)
    async def forward(message_id: str, body: ForwardRequest, request: Request):
        room = request.app.state.room
        user = room.session(request)
        key = room.conversation(user, body.chat_id)
        original = next((item for item in room.messages.get(key, []) if item["id"] == message_id), None)
        if original is None:
            raise HTTPException(404, "Mensagem privada não encontrada nesta conversa.")
        room.check_queue(user)
        # Encaminha só o texto escolhido; não inclui histórico, autor ou identidade do interlocutor.
        message = room.append_message(room.conversation(user, "bot"), user, original["text"])
        room.enqueue(message, user)
        return {"id": message["id"], "chat_id": "bot"}

    app.mount("/assets", StaticFiles(directory=ASSETS), name="assets")
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    for name in ("httpx", "httpx2", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")), access_log=False)
