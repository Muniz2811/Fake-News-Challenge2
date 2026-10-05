"""Estrutura do motor central do Divã Socrático."""

from dataclasses import dataclass, field
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from claims import Alegacao, ExtratorAlegações
from search import Evidencia, ProvedorBuscaWeb
from verifier import VerificadorAlegações, VereditoAlegacao
from socratic import GerenciadorSocratico


@dataclass
class ResultadoAnalise:
    analise_linguistica: str = ""
    alegacoes: list[Alegacao] = field(default_factory=list)
    evidencias: list[Evidencia] = field(default_factory=list)
    vereditos: list[VereditoAlegacao] = field(default_factory=list)
    resultado_factual: str = ""
    requer_revisao: bool = False


class MotorIA:

    def __init__(self):
        self.extrator_claims = ExtratorAlegações()
        self.provedor_busca = ProvedorBuscaWeb()
        self.verificador = VerificadorAlegações()
        self.socratico = GerenciadorSocratico()

    async def analisar_texto(self, texto: str) -> ResultadoAnalise:
        if not texto or not texto.strip():
            raise ValueError("O texto da notícia não pode estar vazio.")

        resultado_claims = self.extrator_claims.extrair(texto)

        todas_evidencias = []
        lista_vereditos = []

        for alegacao in resultado_claims.alegacoes:
            evs = self.provedor_busca.buscar_evidencias_por_alegacao(alegacao.texto)
            todas_evidencias.extend(evs)

            veredito = self.verificador.verificar_alegacao(alegacao, evs)
            lista_vereditos.append(veredito)

        return ResultadoAnalise(
            analise_linguistica="Análise de linguagem concluída.",
            alegacoes=resultado_claims.alegacoes,
            evidencias=todas_evidencias,
            vereditos=lista_vereditos,
            resultado_factual=resultado_claims.para_lista_numerada(),
            requer_revisao=False,
        )