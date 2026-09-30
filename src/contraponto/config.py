"""Carregamento e validação de variáveis de ambiente do Contraponto."""

import logging
import os
from typing import List

logger = logging.getLogger(__name__)


class Config:
    """Centraliza a leitura das variáveis de ambiente necessárias ao bot."""

    def __init__(self) -> None:
        self.telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.authorized_groups: List[int] = self._parse_authorized_groups()

        if not self.telegram_bot_token:
            logger.warning(
                "TELEGRAM_BOT_TOKEN não definido. O bot não conseguirá se conectar."
            )

        if not self.authorized_groups:
            logger.info(
                "AUTHORIZED_GROUPS vazio — o bot aceitará mensagens de qualquer grupo."
            )

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_authorized_groups() -> List[int]:
        raw = os.getenv("AUTHORIZED_GROUPS", "")
        if not raw.strip():
            return []

        ids: List[int] = []
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                ids.append(int(token))
            except ValueError:
                logger.error(
                    "AUTHORIZED_GROUPS contém valor não-numérico: %r — ignorado.",
                    token,
                )
        return ids
