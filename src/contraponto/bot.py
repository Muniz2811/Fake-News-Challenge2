"""Estrutura inicial da interface do Contraponto com o Telegram."""

from .motor_ia import MotorIA


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
