"""Módulo adaptador para busca de evidências web e checagens factuais."""

from dataclasses import dataclass
import os
import sys
import requests

# Garante a localização dos módulos locais
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import GOOGLE_FACT_CHECK_KEY


@dataclass
class Evidencia:
    titulo: str
    url: str
    publicador: str
    trecho: str
    data_publicacao: str = "Data não informada"
    tipo_fonte: str = "secundaria"  # "fact-check", "oficial", "secundaria"


class ProvedorBuscaWeb:
    """Adaptador de busca que converte resultados da Google Fact Check API em evidências padronizadas."""

    def __init__(self, api_key: str = GOOGLE_FACT_CHECK_KEY):
        self.api_key = api_key
        self.base_url_factcheck = (
            "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        )

    def _classificar_fonte(self, url: str, publicador: str) -> str:
        """Classifica a fonte entre fact-check, oficial ou secundária."""
        dominios_oficiais = (".gov.br", ".org.br", ".gov", ".edu", "anvisa.gov.br")
        agencias_factcheck = (
            "lupa",
            "aos fatos",
            "comprova",
            "e-farsas",
            "boatos.org",
            "factcheck",
            "g1.globo.com/fato-ou-fake",
            "fato ou fake",
            "uol confere",
            "estadao",
            "uol",
        )

        url_lower = url.lower()
        pub_lower = publicador.lower()

        if any(ag in url_lower or ag in pub_lower for ag in agencias_factcheck):
            return "fact-check"
        if any(dom in url_lower for dom in dominios_oficiais):
            return "oficial"
        return "secundaria"

    def buscar_evidencias_por_alegacao(
        self, alegacao: str, max_results: int = 3
    ) -> list[Evidencia]:
        """Recebe UMA alegação por vez e devolve lista de evidências normalizadas."""
        if not alegacao or not alegacao.strip():
            return []

        evidencias: list[Evidencia] = []

        if self.api_key:
            try:
                params = {
                    "query": alegacao,
                    "languageCode": "pt-BR",
                    "key": self.api_key,
                }
                response = requests.get(
                    self.base_url_factcheck, params=params, timeout=5
                )

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
                            review_date = review.get("reviewDate", "Data não disponível")
                            rating = review.get("textualRating", "Checagem registrada")

                            tipo = self._classificar_fonte(url, pub_name)

                            evidencias.append(
                                Evidencia(
                                    titulo=title,
                                    url=url,
                                    publicador=pub_name,
                                    trecho=f"Veredito do Checador: {rating}",
                                    data_publicacao=review_date,
                                    tipo_fonte=tipo,
                                )
                            )
            except Exception:
                pass

        return evidencias