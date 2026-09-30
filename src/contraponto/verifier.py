"""Módulo de verificação conservadora de alegações contra evidências e fact-checks."""

from dataclasses import dataclass, field
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from claims import Alegacao
from config import OPENAI_API_KEY
from openai import OpenAI
from search import Evidencia

CATEGORIAS_VALIDAS = {
    "apoiada",
    "refutada",
    "evidencia_insuficiente",
    "conflitante",
    "nao_verificada",
}


@dataclass
class VereditoAlegacao:
    alegacao_id: int
    texto_alegacao: str
    categoria: str
    justificativa: str
    fontes_citadas: list[str] = field(default_factory=list)


class VerificadorAlegacoes:
    """Verifica cada alegação individualmente de forma conservadora."""

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

    def _verificar_por_regras_fallback(
        self, alegacao: Alegacao, evidencias: list[Evidencia]
    ) -> VereditoAlegacao:
        if not evidencias:
            return VereditoAlegacao(
                alegacao_id=alegacao.id,
                texto_alegacao=alegacao.texto,
                categoria="evidencia_insuficiente",
                justificativa=(
                    "Nenhuma evidência ou checagem factual foi encontrada para"
                    " confirmar esta afirmação. O sistema abstém-se de concluir."
                ),
                fontes_citadas=[],
            )

        vereditos_texto = [e.trecho.lower() for e in evidencias]
        tem_falso = any(
            term in v
            for v in vereditos_texto
            for term in ("falso", "fake", "enganoso", "incorreto", "desmentido")
        )
        tem_verdadeiro = any(
            term in v
            for v in vereditos_texto
            for term in ("fato", "verdadeiro", "comprovado", "correto")
        )

        fontes = [f"{e.publicador}: {e.url}" for e in evidencias if e.url]

        if tem_falso and tem_verdadeiro:
            categoria = "conflitante"
            justificativa = (
                "As fontes consultadas apresentam análises divergentes sobre esta"
                " alegação."
            )
        elif tem_falso:
            categoria = "refutada"
            justificativa = (
                "Checagens registradas indicam que a alegação é falsa ou enganosa."
            )
        elif tem_verdadeiro:
            categoria = "apoiada"
            justificativa = (
                "As evidências e checagens encontradas sustentam a veracidade da"
                " afirmação."
            )
        else:
            categoria = "nao_verificada"
            justificativa = (
                "Foram encontrados resultados relacionados, mas sem veredito"
                " conclusivo explícito."
            )

        return VereditoAlegacao(
            alegacao_id=alegacao.id,
            texto_alegacao=alegacao.texto,
            categoria=categoria,
            justificativa=justificativa,
            fontes_citadas=fontes,
        )

    def verificar_alegacao(
        self, alegacao: Alegacao, evidencias: list[Evidencia]
    ) -> VereditoAlegacao:
        if not evidencias:
            return VereditoAlegacao(
                alegacao_id=alegacao.id,
                texto_alegacao=alegacao.texto,
                categoria="evidencia_insuficiente",
                justificativa=(
                    "Ausência de evidências encontradas nas buscas. Abstenção"
                    " conservadora aplicada."
                ),
                fontes_citadas=[],
            )

        if not self.client:
            return self._verificar_por_regras_fallback(alegacao, evidencias)

        contexto_evidencias = "\n".join([
            f"- Fonte: {e.publicador} ({e.tipo_fonte}) | Veredito/Trecho: {e.trecho}"
            f" | Link: {e.url}"
            for e in evidencias
        ])

        prompt_sistema = (
            "Você é um avaliador factual altamente conservador.\n"
            "Sua função é comparar UMA alegação com as evidências fornecidas e"
            " determinar a categoria estrita.\n"
            "Regras Inflexíveis:\n"
            "1. A categoria DEVE ser exatamente uma destas 5: 'apoiada',"
            " 'refutada', 'evidencia_insuficiente', 'conflitante',"
            " 'nao_verificada'.\n"
            "2. Se as evidências se contradizem, use 'conflitante'. NUNCA force um"
            " veredito binário.\n"
            "3. Se as evidências forem fracas ou ausentes, use"
            " 'evidencia_insuficiente' ou 'nao_verificada'.\n"
            "4. NUNCA invente fatos ou URLs que não estejam no contexto.\n"
            "5. A justificativa DEVE citar explicitamente as fontes e evidências"
            " utilizadas.\n"
            "6. Responda ESTRITAMENTE em formato JSON:\n"
            '{\n  "categoria": "categoria_exata",\n  "justificativa": "explicacao'
            ' citando fontes",\n  "fontes_citadas": ["url1", "url2"]\n}'
        )

        prompt_usuario = (
            f"ALEGAÇÃO PARA VERIFICAR:\n{alegacao.texto}\n\n"
            f"EVIDÊNCIAS ENCONTRADAS:\n{contexto_evidencias}"
        )

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": prompt_sistema},
                    {"role": "user", "content": prompt_usuario},
                ],
                temperature=0.0,
            )

            dados = json.loads(response.choices[0].message.content.strip())
            cat = dados.get("categoria", "").lower()

            if cat not in CATEGORIAS_VALIDAS:
                cat = "nao_verificada"

            return VereditoAlegacao(
                alegacao_id=alegacao.id,
                texto_alegacao=alegacao.texto,
                categoria=cat,
                justificativa=dados.get("justificativa", ""),
                fontes_citadas=dados.get("fontes_citadas", []),
            )
        except Exception:
            return self._verificar_por_regras_fallback(alegacao, evidencias)