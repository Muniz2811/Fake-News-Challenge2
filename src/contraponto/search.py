"""Módulo adaptador para busca de evidências web e checagens factuais."""

import logging
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from urllib.parse import urlparse

import requests

# Garante a localização dos módulos locais
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import GOOGLE_FACT_CHECK_KEY, TAVILY_API_KEY

logger = logging.getLogger(__name__)


@dataclass
class Evidencia:
    """Modelo normalizado de evidência retornada por qualquer provedor de busca."""

    title: str
    url: str
    publisher: str
    snippet: str
    published_at: str | None = None
    relevance_score: float = 0.0
    source_type: str = "unknown"  # "official", "fact_check", "news", "secondary", "unknown"


# ------------------------------------------------------------------ #
#  Listas de domínios para classificação objetiva de fonte            #
# ------------------------------------------------------------------ #

_DOMINIOS_OFICIAIS = (".gov.br", ".gov", ".edu", ".edu.br", ".org.br")

_AGENCIAS_FACTCHECK = (
    "lupa",
    "aos fatos",
    "aosfatos",
    "comprova",
    "e-farsas",
    "boatos.org",
    "factcheck",
    "fato-ou-fake",
    "fato ou fake",
    "uol confere",
    "snopes",
    "politifact",
    "checamos",
)

_DOMINIOS_NEWS = (
    "g1.globo.com",
    "uol.com.br",
    "folha.uol.com.br",
    "estadao.com.br",
    "bbc.com",
    "bbc.co.uk",
    "reuters.com",
    "apnews.com",
    "cnn.com",
    "nytimes.com",
)


def classificar_fonte(url: str, publisher: str = "") -> str:
    """Classifica a natureza da fonte de forma objetiva.

    Categorias: ``official``, ``fact_check``, ``news``, ``secondary``, ``unknown``.
    """
    url_lower = url.lower()
    pub_lower = publisher.lower()

    if any(ag in url_lower or ag in pub_lower for ag in _AGENCIAS_FACTCHECK):
        return "fact_check"
    if any(dom in url_lower for dom in _DOMINIOS_OFICIAIS):
        return "official"
    if any(dom in url_lower for dom in _DOMINIOS_NEWS):
        return "news"

    # Se possui domínio reconhecível, mas não enquadrado acima → secundária
    parsed = urlparse(url)
    if parsed.netloc:
        return "secondary"

    return "unknown"


# ------------------------------------------------------------------ #
#  Interface abstrata de provedor de busca                            #
# ------------------------------------------------------------------ #


class WebSearchProvider(ABC):
    """Contrato para qualquer provedor de busca web de evidências."""

    @abstractmethod
    def buscar_evidencias_por_alegacao(
        self, alegacao: str, max_results: int = 3
    ) -> list[Evidencia]:
        """Recebe UMA alegação isolada e devolve evidências normalizadas."""


# ------------------------------------------------------------------ #
#  Google Fact Check Adapter                                          #
# ------------------------------------------------------------------ #


class GoogleFactCheckAdapter(WebSearchProvider):
    """Adaptador que converte resultados da Google Fact Check API em evidências padronizadas."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key if api_key is not None else GOOGLE_FACT_CHECK_KEY
        self.base_url = (
            "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        )

    def buscar_evidencias_por_alegacao(
        self, alegacao: str, max_results: int = 3
    ) -> list[Evidencia]:
        """Recebe UMA alegação por vez e devolve lista de evidências normalizadas.

        Garante degradação graciosa em caso de indisponibilidade, erros ou falta de chave.
        """
        if not alegacao or not alegacao.strip():
            return []

        evidencias: list[Evidencia] = []

        if not self.api_key:
            logger.warning("GOOGLE_FACT_CHECK_KEY ausente — provedor Google desabilitado.")
            return evidencias

        try:
            params = {
                "query": alegacao,
                "languageCode": "pt-BR",
                "key": self.api_key,
            }
            response = requests.get(self.base_url, params=params, timeout=5)

            if response.status_code == 200:
                data = response.json()
                claims = data.get("claims", [])

                for item in claims[:max_results]:
                    text_claim = item.get("text", alegacao)
                    reviews = item.get("claimReview", [])

                    for review in reviews:
                        publisher_info = review.get("publisher", {})
                        pub_name = publisher_info.get("name", "Fonte Desconhecida")
                        title = review.get("title") or f"Checagem sobre: {text_claim}"
                        url = review.get("url", "")
                        review_date = review.get("reviewDate") or None
                        rating = review.get("textualRating", "Checagem registrada")

                        tipo = classificar_fonte(url, pub_name)

                        evidencias.append(
                            Evidencia(
                                title=title,
                                url=url,
                                publisher=pub_name,
                                snippet=f"Veredito do Checador: {rating}",
                                published_at=review_date,
                                relevance_score=0.0,
                                source_type=tipo,
                            )
                        )
            else:
                logger.warning(
                    "Google Fact Check retornou status %s para alegação: %.80s",
                    response.status_code,
                    alegacao,
                )
        except Exception:
            # Indisponibilidade do provedor não interrompe o fluxo principal
            logger.exception("Erro ao consultar Google Fact Check API.")

        return evidencias


# ------------------------------------------------------------------ #
#  Tavily Search Adapter                                              #
# ------------------------------------------------------------------ #

_TAVILY_SEARCH_URL = "https://api.tavily.com/search"


class TavilySearchAdapter(WebSearchProvider):
    """Adaptador que consulta a Tavily Search API e normaliza resultados em Evidencia."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key if api_key is not None else TAVILY_API_KEY

    def buscar_evidencias_por_alegacao(
        self, alegacao: str, max_results: int = 5
    ) -> list[Evidencia]:
        """Recebe UMA alegação isolada e devolve evidências normalizadas da Tavily.

        Garante degradação graciosa: se a chave estiver ausente, a API retornar
        erro, houver timeout ou qualquer exceção, retorna lista vazia sem
        interromper o pipeline.
        """
        if not alegacao or not alegacao.strip():
            return []

        if not self.api_key:
            logger.warning("TAVILY_API_KEY ausente — provedor Tavily desabilitado.")
            return []

        try:
            payload = {
                "api_key": self.api_key,
                "query": alegacao,
                "max_results": max_results,
                "include_answer": False,
            }
            response = requests.post(
                _TAVILY_SEARCH_URL, json=payload, timeout=10
            )

            if response.status_code != 200:
                logger.warning(
                    "Tavily retornou status %s para alegação: %.80s",
                    response.status_code,
                    alegacao,
                )
                return []

            data = response.json()
            results = data.get("results", [])

            evidencias: list[Evidencia] = []
            for item in results[:max_results]:
                url = item.get("url", "")
                raw_publisher = self._extrair_publisher(url)
                published_at = item.get("published_date") or None

                evidencias.append(
                    Evidencia(
                        title=item.get("title", ""),
                        url=url,
                        publisher=raw_publisher,
                        snippet=item.get("content", ""),
                        published_at=published_at,
                        relevance_score=float(item.get("score", 0.0)),
                        source_type=classificar_fonte(url, raw_publisher),
                    )
                )

            return evidencias

        except requests.exceptions.Timeout:
            logger.warning("Timeout ao consultar Tavily para: %.80s", alegacao)
        except requests.exceptions.ConnectionError:
            logger.warning("Erro de conexão ao consultar Tavily.")
        except Exception:
            logger.exception("Erro inesperado ao consultar Tavily API.")

        return []

    @staticmethod
    def _extrair_publisher(url: str) -> str:
        """Extrai o domínio principal da URL como nome do publicador."""
        try:
            parsed = urlparse(url)
            netloc = parsed.netloc
            if netloc.startswith("www."):
                netloc = netloc[4:]
            return netloc or "Fonte Desconhecida"
        except Exception:
            return "Fonte Desconhecida"


# ------------------------------------------------------------------ #
#  Alias de compatibilidade                                           #
# ------------------------------------------------------------------ #

# Mantém compatibilidade com imports existentes em motor_ia.py e bot.py
ProvedorBuscaWeb = GoogleFactCheckAdapter