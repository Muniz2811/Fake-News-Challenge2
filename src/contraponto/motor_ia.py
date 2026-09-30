"""Estrutura inicial do motor central do Divã Socrático."""

from dataclasses import dataclass, field


@dataclass
class ResultadoAnalise:
    """Mantém análise linguística e resultado factual separados."""

    analise_linguistica: str = ""
    alegacoes: list[str] = field(default_factory=list)
    resultado_factual: str = ""
    fontes: list[str] = field(default_factory=list)
    requer_revisao: bool = False


class MotorIA:
    """Contrato inicial para os componentes que serão criados pelas issues do MVP."""

    async def analisar_texto(self, texto: str) -> ResultadoAnalise:
        if not texto or not texto.strip():
            raise ValueError("O texto da notícia não pode estar vazio.")

        # TODO: integrar análise linguística, extração de alegações,
        # busca de evidências, verificação e diálogo socrático.
        raise NotImplementedError(
            "O fluxo de análise será implementado pelas próximas issues do MVP."
        )