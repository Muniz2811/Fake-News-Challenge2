"""Estrutura do motor central do Divã Socrático."""

from dataclasses import dataclass, field
from claims import ExtratorAlegações, Alegacao
from search import ProvedorBuscaWeb, Evidencia


@dataclass
class ResultadoAnalise:
  """Mantém análise linguística, alegações e evidências separadas."""

  analise_linguistica: str = ""
  alegacoes: list[Alegacao] = field(default_factory=list)
  evidencias: list[Evidencia] = field(default_factory=list)
  resultado_factual: str = ""
  requer_revisao: bool = False


class MotorIA:

  def __init__(self):
    self.extrator_claims = ExtratorAlegações()
    self.provedor_busca = ProvedorBuscaWeb()

  async def analisar_texto(self, texto: str) -> ResultadoAnalise:
    if not texto or not texto.strip():
      raise ValueError("O texto da notícia não pode estar vazio.")

    # 1. Extração de alegações (MVP-06)
    resultado_claims = self.extrator_claims.extrair(texto)

    # 2. Busca de evidências por alegação individual (MVP-09)
    todas_evidencias = []
    for alegacao in resultado_claims.alegacoes:
      evs = self.provedor_busca.buscar_evidencias_por_alegacao(alegacao.texto)
      todas_evidencias.extend(evs)

    return ResultadoAnalise(
        analise_linguistica="Análise inicial de linguagem concluída.",
        alegacoes=resultado_claims.alegacoes,
        evidencias=todas_evidencias,
        resultado_factual=resultado_claims.para_lista_numerada(),
        requer_revisao=False,
    )