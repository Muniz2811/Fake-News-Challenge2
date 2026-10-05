"""Motor central: extração, busca por provedor e verificação por alegação."""
import asyncio
import logging
import os
import sys
from dataclasses import dataclass, field

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from claims import Alegacao, ExtratorAlegações
from config import LANGUAGE_RISK_MODEL_PATH
from language_risk import AnalisadorLinguistico, formatar_analise_linguistica
from search import Evidencia, GoogleFactCheckAdapter, TavilySearchAdapter, WebSearchProvider
from verifier import ROTULOS_CATEGORIAS, VereditoAlegacao, VerificadorAlegacoes

logger = logging.getLogger(__name__)

_NOMES_PROVEDORES = {
    "GoogleFactCheckAdapter": "Google Fact Check",
    "TavilySearchAdapter": "Tavily",
}


@dataclass
class ResultadoAnalise:
    """Mantém linguagem, alegações, evidências e conclusões individuais separadas."""

    analise_linguistica: str = ""
    alegacoes: list[Alegacao] = field(default_factory=list)
    evidencias: list[Evidencia] = field(default_factory=list)
    resultado_factual: str = ""
    requer_revisao: bool = False
    vereditos: list[VereditoAlegacao] = field(default_factory=list)
    score_linguistico: float | None = None
    etapas_indisponiveis: list[str] = field(default_factory=list)


class MotorIA:
    def __init__(
        self, provedores_busca: list[WebSearchProvider] | None = None,
        verificador: VerificadorAlegacoes | None = None,
        analisador_linguistico: AnalisadorLinguistico | None = None,
    ):
        self.extrator_claims = ExtratorAlegações()
        self.provedores_busca = provedores_busca if provedores_busca is not None else [
            GoogleFactCheckAdapter(), TavilySearchAdapter(),
        ]
        self.verificador = verificador if verificador is not None else VerificadorAlegacoes()
        self.analisador_linguistico = (
            analisador_linguistico if analisador_linguistico is not None
            else AnalisadorLinguistico(LANGUAGE_RISK_MODEL_PATH)
        )

    async def analisar_texto(self, texto: str) -> ResultadoAnalise:
        if not texto or not texto.strip():
            raise ValueError("O texto da notícia não pode estar vazio.")
        score = await asyncio.to_thread(self.analisador_linguistico.analisar, texto)
        extracao = self.extrator_claims.extrair(texto)
        evidencias = []
        vereditos = []
        indisponiveis: list[str] = []
        for alegacao in extracao.alegacoes:
            individuais = []
            for provedor in self.provedores_busca:
                try:
                    individuais.extend(provedor.buscar_evidencias_por_alegacao(alegacao.texto))
                except Exception:
                    logger.warning("Provedor %s indisponível.", type(provedor).__name__)
                    nome = _NOMES_PROVEDORES.get(type(provedor).__name__, type(provedor).__name__)
                    aviso = f"Fonte indisponível: a busca no {nome} não respondeu."
                    if aviso not in indisponiveis:
                        indisponiveis.append(aviso)
            evidencias.extend(individuais)
            vereditos.append(self.verificador.verificar_alegacao(alegacao, individuais))
        factual = "\n\n".join(
            f"{v.alegacao_id}. {v.texto_alegacao}\n"
            f"Conclusão: {ROTULOS_CATEGORIAS[v.categoria]}\n{v.justificativa}"
            for v in vereditos
        ) or "Nenhuma alegação verificável foi identificada no texto."
        return ResultadoAnalise(
            analise_linguistica=formatar_analise_linguistica(
                score, self.analisador_linguistico.indisponibilidade,
            ),
            alegacoes=extracao.alegacoes, evidencias=evidencias, vereditos=vereditos,
            resultado_factual=factual, requer_revisao=False,
            score_linguistico=score, etapas_indisponiveis=indisponiveis,
        )
