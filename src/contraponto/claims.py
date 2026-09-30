"""Módulo responsável por extrair e isolar alegações verificáveis de textos em português."""

import json
import os
import re
import sys
from dataclasses import dataclass, field

# Garante que o diretório 'src/contraponto' seja localizado para importação de módulos irmãos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import MAX_CLAIMS_PER_MESSAGE, OPENAI_API_KEY
from openai import OpenAI


@dataclass
class Alegacao:
    id: int
    texto: str
    trecho_origem: str = ""
    posicao: int = 0


@dataclass
class ResultadoExtracao:
    texto_original: str
    alegacoes: list[Alegacao] = field(default_factory=list)

    def para_lista_numerada(self) -> str:
        """Retorna as alegações formatadas em lista numerada."""
        if not self.alegacoes:
            return "Nenhuma alegação verificável foi identificada no texto."
        return "\n".join([f"{a.id}. {a.texto}" for a in self.alegacoes])


class ExtratorAlegações:
    """Extrai afirmações factuais checáveis evitando opiniões e perguntas."""

    def __init__(self, limit: int = MAX_CLAIMS_PER_MESSAGE):
        self.limit = limit
        self.client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

    def _extrair_por_regras_fallback(self, texto: str) -> list[Alegacao]:
        """Extração baseada em heurísticas e regras para fallback sem API key."""
        # Captura frases mantendo a pontuação final para detetar '?'
        frases = [f.strip() for f in re.findall(r'[^.!?]+[.!?]?', texto) if f.strip()]
        alegacoes = []
        contador = 1

        gatilhos_pergunta = ("será", "como", "por que", "porque", "qual", "quais", "quem", "onde", "quando")

        for frase in frases:
            if contador > self.limit:
                break

            frase_limpa = frase.rstrip(".!? ").strip()

            # Descarta perguntas (com '?' ou iniciadas por palavras interrogativas), frases curtas e opiniões
            if frase.endswith("?") or frase_limpa.lower().startswith(gatilhos_pergunta) or len(frase_limpa.split()) < 4:
                continue
            if re.search(r"\b(eu acho|na minha opinião|infelizmente|talvez)\b", frase_limpa, re.IGNORECASE):
                continue

            alegacoes.append(
                Alegacao(
                    id=contador,
                    texto=frase_limpa,
                    trecho_origem=frase_limpa,
                    posicao=texto.find(frase_limpa)
                )
            )
            contador += 1

        return alegacoes

    def extrair(self, texto: str) -> ResultadoExtracao:
        """Extrai até `self.limit` alegações verificáveis do texto informado."""
        if not texto or not texto.strip():
            return ResultadoExtracao(texto_original=texto, alegacoes=[])

        if not self.client:
            return ResultadoExtracao(
                texto_original=texto,
                alegacoes=self._extrair_por_regras_fallback(texto),
            )

        prompt_sistema = (
            "Você é um assistente especializado em checagem de fatos em português.\n"
            "Sua tarefa é extrair APENAS afirmações factuais concretas e"
            " verificáveis contidas no texto.\n"
            "Diretrizes estritas:\n"
            "1. Ignore perguntas, opiniões, sentimentos pessoais e julgamentos de"
            " valor.\n"
            f"2. Extraia no máximo {self.limit} afirmações centrais.\n"
            "3. Preserve o sentido original e inclua o trecho de origem exato.\n"
            "4. Responda ESTRITAMENTE em formato JSON no seguinte padrão:\n"
            '{\n  "alegacoes": [\n    {"texto": "afirmação imparcial",'
            ' "trecho_origem": "trecho exato"}\n  ]\n}'
        )

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": prompt_sistema},
                    {"role": "user", "content": texto},
                ],
                temperature=0.0,
            )

            conteudo = response.choices[0].message.content.strip()
            dados = json.loads(conteudo)

            alegacoes_processadas = []
            for idx, item in enumerate(
                dados.get("alegacoes", [])[: self.limit], start=1
            ):
                txt_alegacao = item.get("texto", "").strip()
                origem = item.get("trecho_origem", "").strip()
                if txt_alegacao:
                    alegacoes_processadas.append(
                        Alegacao(
                            id=idx,
                            texto=txt_alegacao,
                            trecho_origem=origem,
                            posicao=texto.find(origem) if origem in texto else 0,
                        )
                    )

            return ResultadoExtracao(
                texto_original=texto, alegacoes=alegacoes_processadas
            )

        except (json.JSONDecodeError, KeyError, Exception):
            return ResultadoExtracao(
                texto_original=texto,
                alegacoes=self._extrair_por_regras_fallback(texto),
            )