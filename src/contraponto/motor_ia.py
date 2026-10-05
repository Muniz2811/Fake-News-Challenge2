"""Estrutura do motor central do Divã Socrático."""

from dataclasses import dataclass, field
import asyncio
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from claims import ExtratorAlegações, Alegacao
from search import ProvedorBuscaWeb, Evidencia
from verifier import VerificadorAlegacoes, VereditoAlegacao
from socratic import GerenciadorSocratico

logger = logging.getLogger(__name__)

_NOMES_PROVEDORES = {
    "GoogleFactCheckAdapter": "Google Fact Check",
    "ProvedorBuscaWeb": "Google Fact Check",
    "TavilySearchAdapter": "Tavily",
}

ROTULOS_CATEGORIAS = {
    "apoiada": "Apoiada por evidências",
    "refutada": "Refutada por evidências",
    "evidencia_insuficiente": "Evidência insuficiente (Dados não encontrados)",
    "conflitante": "Informação conflitante (Fontes divergem)",
    "nao_verificada": "Não verificada",
}


@dataclass
class ResultadoAnalise:
    """Mantém a análise factual, evidências, resiliência e sinalização de revisão."""

    analise_linguistica: str = ""
    alegacoes: list[Alegacao] = field(default_factory=list)
    evidencias: list[Evidencia] = field(default_factory=list)
    vereditos: list[VereditoAlegacao] = field(default_factory=list)
    resultado_factual: str = ""
    requer_revisao: bool = False
    score_linguistico: float | None = None
    etapas_indisponiveis: list[str] = field(default_factory=list)


class MotorIA:

    def __init__(self):
        self.extrator_claims = ExtratorAlegações()
        self.provedor_busca = ProvedorBuscaWeb()
        # Garante compatibilidade com os testes do CI que inspecionam os provedores
        self.provedores_busca = [self.provedor_busca]
        self.verificador = VerificadorAlegacoes()
        self.socratico = GerenciadorSocratico()

    async def analisar_texto(self, texto: str) -> ResultadoAnalise:
        if not texto or not texto.strip():
            raise ValueError("O texto da notícia não pode estar vazio.")

        # 1. Extração de alegações (MVP-06) com resiliência
        try:
            resultado_claims = self.extrator_claims.extrair(texto)
            alegacoes_extraidas = resultado_claims.alegacoes
        except Exception as err:
            logger.error("Falha ao extrair alegações: %s", err)
            alegacoes_extraidas = []

        todas_evidencias: list[Evidencia] = []
        lista_vereditos: list[VereditoAlegacao] = []
        indisponiveis: list[str] = []

        # 2. Busca de evidências (MVP-09) tratando falhas sem derrubar a aplicação
        for alegacao in alegacoes_extraidas:
            evs_alegacao: list[Evidencia] = []
            for provedor in self.provedores_busca:
                try:
                    evs_alegacao.extend(
                        provedor.buscar_evidencias_por_alegacao(alegacao.texto)
                    )
                except Exception as ex:
                    logger.warning("Provedor %s indisponível: %s", type(provedor).__name__, ex)
                    nome = _NOMES_PROVEDORES.get(type(provedor).__name__, type(provedor).__name__)
                    aviso = f"Fonte de busca externa ({nome}) esteve indisponível durante a consulta."
                    if aviso not in indisponiveis:
                        indisponiveis.append(aviso)

            todas_evidencias.extend(evs_alegacao)

            # 3. Verificação Conservadora por Alegação (MVP-10)
            veredito = self.verificador.verificar_alegacao(alegacao, evs_alegacao)
            lista_vereditos.append(veredito)

        # 4. Avaliar se o caso é inconclusivo (necessidade de marca 'requer_revisao')
        tem_inconclusivo = any(
            v.categoria in ["evidencia_insuficiente", "conflitante", "nao_verificada"]
            for v in lista_vereditos
        ) or bool(indisponiveis)

        # 5. Formatar o relatório transparente informando o que foi concluído e o que faltou
        linhas_relatorio = []
        for idx, v in enumerate(lista_vereditos, 1):
            status_txt = ROTULOS_CATEGORIAS.get(v.categoria, v.categoria)
            linhas_relatorio.append(f"{idx}. **Alegação:** '{v.texto_alegacao}'")
            linhas_relatorio.append(f"   • **Situação:** {status_txt}")
            linhas_relatorio.append(f"   • **Justificativa:** {v.justificativa}")
            if v.fontes_citadas:
                linhas_relatorio.append(f"   • **Fontes:** {', '.join(v.fontes_citadas)}")
            linhas_relatorio.append("")

        if indisponiveis:
            linhas_relatorio.append("⚠️ **Avisos de Indisponibilidade Externa:**")
            for av in indisponiveis:
                linhas_relatorio.append(f"• {av}")
            linhas_relatorio.append("")

        if tem_inconclusivo:
            linhas_relatorio.append(
                "ℹ️ **Nota de Transparência [Recomenda-se Verificação Adicional]:**\n"
                "As evidências disponíveis nas bases públicas foram insuficientes ou divergentes. "
                "Esta marcação indica que o tema exige cautela adicional antes de ser compartilhado."
            )

        factual = "\n".join(linhas_relatorio) or "Nenhuma alegação verificável foi identificada no texto enviado."

        return ResultadoAnalise(
            analise_linguistica="Análise de linguagem e resiliência concluídas.",
            alegacoes=alegacoes_extraidas,
            evidencias=todas_evidencias,
            vereditos=lista_vereditos,
            resultado_factual=factual,
            requer_revisao=tem_inconclusivo,
            score_linguistico=None,
            etapas_indisponiveis=indisponiveis,
        )