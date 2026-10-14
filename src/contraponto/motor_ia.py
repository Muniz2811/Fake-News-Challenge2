"""Estrutura do motor central do Divã Socrático."""

from dataclasses import dataclass, field
import asyncio
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from claims import ExtratorAlegações, Alegacao
from search import GoogleFactCheckAdapter, TavilySearchAdapter, WebSearchProvider, Evidencia
from config import (
    BERTIMBAU_DEVICE,
    BERTIMBAU_LOGREG_METRICS_PATH,
    BERTIMBAU_MLP_METRICS_PATH,
    BERTIMBAU_MODEL_DIR,
    LANGUAGE_RISK_BACKEND,
    LANGUAGE_RISK_MODEL_PATH,
)
from language_risk import (
    AnalisadorBertimbauMLP,
    AnalisadorBertimbauLogReg,
    AnalisadorLinguistico,
    formatar_analise_linguistica,
)
from verifier import VerificadorAlegacoes, VereditoAlegacao, ROTULOS_CATEGORIAS as ROTULOS_FACTUAIS
from socratic import GerenciadorSocratico
from llm import ProvedorLLM, criar_provedor_llm
from diagnostics import log_stage, logged_async_stage

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

    texto_original: str = ""
    analise_linguistica: str = ""
    alegacoes: list[Alegacao] = field(default_factory=list)
    evidencias: list[Evidencia] = field(default_factory=list)
    vereditos: list[VereditoAlegacao] = field(default_factory=list)
    resultado_factual: str = ""
    requer_revisao: bool = False
    score_linguistico: float | None = None
    etapas_indisponiveis: list[str] = field(default_factory=list)


class MotorIA:

    def __init__(
        self, provedores_busca: list[WebSearchProvider] | None = None,
        verificador: VerificadorAlegacoes | None = None,
        analisador_linguistico: AnalisadorLinguistico | None = None,
        llm: ProvedorLLM | None = None,
        extrator_claims: ExtratorAlegações | None = None,
        gerenciador_socratico: GerenciadorSocratico | None = None,
    ):
        provedor_llm = llm if llm is not None else criar_provedor_llm()
        self.extrator_claims = (
            extrator_claims if extrator_claims is not None
            else ExtratorAlegações(llm=provedor_llm)
        )
        self.provedores_busca = provedores_busca if provedores_busca is not None else [
            GoogleFactCheckAdapter(), TavilySearchAdapter(),
        ]
        self.provedor_busca = self.provedores_busca[0] if self.provedores_busca else None
        self.verificador = (
            verificador if verificador is not None
            else VerificadorAlegacoes(llm=provedor_llm)
        )
        if analisador_linguistico is not None:
            self.analisador_linguistico = analisador_linguistico
        elif LANGUAGE_RISK_BACKEND == "bertimbau_mlp":
            self.analisador_linguistico = AnalisadorBertimbauMLP(
                BERTIMBAU_MLP_METRICS_PATH,
                BERTIMBAU_MODEL_DIR,
                BERTIMBAU_DEVICE,
            )
        elif LANGUAGE_RISK_BACKEND == "bertimbau_logreg":
            self.analisador_linguistico = AnalisadorBertimbauLogReg(
                BERTIMBAU_LOGREG_METRICS_PATH,
                BERTIMBAU_MODEL_DIR,
                BERTIMBAU_DEVICE,
            )
        else:
            self.analisador_linguistico = AnalisadorLinguistico(
                LANGUAGE_RISK_MODEL_PATH
            )
        self.socratico = (
            gerenciador_socratico if gerenciador_socratico is not None
            else GerenciadorSocratico(llm=provedor_llm)
        )

    @logged_async_stage("analysis_pipeline")
    async def analisar_texto(self, texto: str) -> ResultadoAnalise:
        if not texto or not texto.strip():
            raise ValueError("O texto da notícia não pode estar vazio.")

        indisponiveis: list[str] = []
        score = None
        motivo_linguistico = self.analisador_linguistico.indisponibilidade
        try:
            with log_stage(logger, "language_analysis"):
                score = await asyncio.to_thread(self.analisador_linguistico.analisar, texto)
            motivo_linguistico = self.analisador_linguistico.indisponibilidade
        except Exception as err:
            logger.error("error_category=INTERNAL_ERROR stage=language_analysis error_type=%s", type(err).__name__)
            motivo_linguistico = "falha na inferência"
            indisponiveis.append("Análise linguística indisponível nesta investigação.")

        # 1. Extração de alegações (MVP-06) com resiliência
        try:
            with log_stage(logger, "claim_extraction"):
                resultado_claims = await asyncio.to_thread(self.extrator_claims.extrair, texto)
            alegacoes_extraidas = resultado_claims.alegacoes
        except Exception as err:
            logger.error("error_category=INTERNAL_ERROR stage=claim_extraction error_type=%s", type(err).__name__)
            alegacoes_extraidas = []
            indisponiveis.append("Extração de alegações indisponível nesta investigação.")

        todas_evidencias: list[Evidencia] = []
        lista_vereditos: list[VereditoAlegacao] = []

        # 2. Busca de evidências (MVP-09) tratando falhas sem derrubar a aplicação
        for alegacao in alegacoes_extraidas:
            evs_alegacao: list[Evidencia] = []
            for provedor in self.provedores_busca:
                try:
                    with log_stage(logger, "evidence_search", error_category="EXTERNAL_SERVICE_UNAVAILABLE"):
                        evs_alegacao.extend(await asyncio.to_thread(
                            provedor.buscar_evidencias_por_alegacao, alegacao.texto,
                        ))
                    # Adaptadores preservam lista vazia em falhas, mas sinalizam a causa.
                    erro = getattr(provedor, "ultimo_erro", None)
                    if isinstance(erro, str) and erro:
                        nome = _NOMES_PROVEDORES.get(type(provedor).__name__, type(provedor).__name__)
                        aviso = f"Fonte indisponível: a busca no {nome} não respondeu."
                        if aviso not in indisponiveis:
                            indisponiveis.append(aviso)
                except Exception:
                    nome = _NOMES_PROVEDORES.get(type(provedor).__name__, type(provedor).__name__)
                    aviso = f"Fonte indisponível: a busca no {nome} não respondeu."
                    if aviso not in indisponiveis:
                        indisponiveis.append(aviso)

            todas_evidencias.extend(evs_alegacao)

            # 3. Verificação Conservadora por Alegação (MVP-10)
            try:
                with log_stage(logger, "claim_verification"):
                    veredito = await asyncio.to_thread(
                        self.verificador.verificar_alegacao, alegacao, evs_alegacao,
                    )
            except Exception as err:
                logger.error("error_category=INTERNAL_ERROR stage=claim_verification error_type=%s", type(err).__name__)
                veredito = VerificadorAlegacoes._abster(
                    alegacao, "Verificação indisponível. Abstenção aplicada.",
                )
                aviso = "Verificação factual indisponível nesta investigação."
                if aviso not in indisponiveis:
                    indisponiveis.append(aviso)
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
            linhas_relatorio.append(f"Conclusão: {ROTULOS_FACTUAIS[v.categoria]}")
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
            texto_original=texto,
            analise_linguistica=formatar_analise_linguistica(
                score,
                motivo_linguistico,
                getattr(
                    self.analisador_linguistico,
                    "descricao_modelo",
                    "TF-IDF + regressão logística",
                ),
            ),
            alegacoes=alegacoes_extraidas,
            evidencias=todas_evidencias,
            vereditos=lista_vereditos,
            resultado_factual=factual,
            requer_revisao=tem_inconclusivo,
            score_linguistico=score,
            etapas_indisponiveis=indisponiveis,
        )
