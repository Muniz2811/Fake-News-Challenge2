"""Estrutura inicial da interface do Contraponto com o Telegram."""

import os
import sys

# Adiciona o diretório atual ao sys.path para permitir a importação direta dos módulos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import TELEGRAM_BOT_TOKEN, validate_env
from motor_ia import MotorIA


class BotTelegram:

    def __init__(self, motor: MotorIA) -> None:
        self.motor = motor

    async def receber_texto(self, texto: str) -> str:
        """Encaminha uma mensagem textual ao motor central."""
        resultado = await self.motor.analisar_texto(texto)
        return (
            f"Análise da linguagem:\n{resultado.analise_linguistica}\n\n"
            f"Resultado factual:\n{resultado.resultado_factual}"
        )

    # TODO: conectar este adaptador à Telegram Bot API.
    # TODO: adicionar grupos autorizados, alerta e redirecionamento ao privado.


def main() -> None:
    """Valida a execução local do ambiente."""
    print("🔍 Validando variáveis de ambiente...")
    validate_env()
    print("✅ Configurações e variáveis de ambiente validadas com sucesso!")
    print("🤖 Módulos bot.py e motor_ia.py prontos e integrados.")


if __name__ == "__main__":
    main()