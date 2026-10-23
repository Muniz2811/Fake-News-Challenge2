"""Testes automatizados para os adaptadores de busca web (Google Fact Check e Tavily).

Os mocks de módulos (requests, dotenv) são aplicados antes da importação para
permitir a execução dos testes em ambientes sem as dependências instaladas.
"""

import os
import sys
import types
import unittest
from unittest.mock import patch, MagicMock

# ------------------------------------------------------------------ #
#  Stub de módulos externos que podem estar ausentes no ambiente CI   #
# ------------------------------------------------------------------ #

# Garante que 'requests' exista como módulo (pode ser real ou stub)
if "requests" not in sys.modules:
    _requests_stub = types.ModuleType("requests")

    class _FakeResponse:
        status_code = 200
        def json(self):
            return {}

    _requests_stub.get = MagicMock(return_value=_FakeResponse())
    _requests_stub.post = MagicMock(return_value=_FakeResponse())

    _exceptions = types.ModuleType("requests.exceptions")
    _exceptions.Timeout = type("Timeout", (IOError,), {})
    _exceptions.ConnectionError = type("ConnectionError", (IOError,), {})
    _exceptions.RequestException = type("RequestException", (IOError,), {})
    _requests_stub.exceptions = _exceptions
    sys.modules["requests"] = _requests_stub
    sys.modules["requests.exceptions"] = _exceptions

# Garante que 'dotenv' exista
if "dotenv" not in sys.modules:
    _dotenv_stub = types.ModuleType("dotenv")
    _dotenv_stub.load_dotenv = lambda *a, **kw: None
    sys.modules["dotenv"] = _dotenv_stub

# Garante que 'openai' exista (importado indiretamente por claims.py → motor_ia.py)
if "openai" not in sys.modules:
    _openai_stub = types.ModuleType("openai")
    _openai_stub.OpenAI = MagicMock()
    sys.modules["openai"] = _openai_stub

# Adiciona o diretório de código-fonte ao sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "contraponto"))

from search import (
    Evidencia,
    TavilySearchAdapter,
    GoogleFactCheckAdapter,
    WebSearchProvider,
    classificar_fonte,
)


class TestEvidenciaModel(unittest.TestCase):
    """Verifica o modelo de dados normalizado."""

    def test_evidencia_campos_obrigatorios(self):
        ev = Evidencia(
            title="Título",
            url="https://example.com",
            publisher="Example",
            snippet="Um trecho",
        )
        self.assertEqual(ev.title, "Título")
        self.assertEqual(ev.url, "https://example.com")
        self.assertEqual(ev.publisher, "Example")
        self.assertEqual(ev.snippet, "Um trecho")
        self.assertIsNone(ev.published_at)
        self.assertEqual(ev.relevance_score, 0.0)
        self.assertEqual(ev.source_type, "unknown")

    def test_published_at_default_none(self):
        """Caso 2: resultado sem data retorna published_at=None."""
        ev = Evidencia(title="T", url="http://x.com", publisher="P", snippet="S")
        self.assertIsNone(ev.published_at)


class TestClassificarFonte(unittest.TestCase):
    """Caso 4: A classificação source_type é assinalada conforme a regra implementada."""

    def test_oficial_gov_br(self):
        self.assertEqual(classificar_fonte("https://www.anvisa.gov.br/nota"), "official")

    def test_oficial_edu(self):
        self.assertEqual(classificar_fonte("https://universidade.edu/artigo"), "official")

    def test_oficial_justica_eleitoral(self):
        self.assertEqual(classificar_fonte("https://www.tse.jus.br/noticia"), "official")

    def test_fact_check_lupa(self):
        self.assertEqual(classificar_fonte("https://lupa.uol.com.br/check"), "fact_check")

    def test_fact_check_snopes(self):
        self.assertEqual(classificar_fonte("https://snopes.com/check"), "fact_check")

    def test_fact_check_por_publisher(self):
        self.assertEqual(classificar_fonte("https://random.com", "Lupa"), "fact_check")

    def test_news_bbc(self):
        self.assertEqual(classificar_fonte("https://bbc.com/news/article"), "news")

    def test_news_dw(self):
        self.assertEqual(classificar_fonte("https://www.dw.com/pt-br/noticia"), "news")

    def test_secondary_generic(self):
        self.assertEqual(classificar_fonte("https://blogdojose.com.br/post"), "secondary")

    def test_unknown_vazio(self):
        self.assertEqual(classificar_fonte(""), "unknown")


# ------------------------------------------------------------------ #
#  Tavily Adapter                                                     #
# ------------------------------------------------------------------ #


_TAVILY_SUCESSO_RESPONSE = {
    "results": [
        {
            "title": "Vacina X é segura, diz OMS",
            "url": "https://who.int/news/vacina-x",
            "content": "A Organização Mundial da Saúde confirmou a segurança da vacina X.",
            "score": 0.87,
            "published_date": "2025-03-15",
        },
        {
            "title": "Fact-check: vacina X não causa infertilidade",
            "url": "https://aosfatos.org/check/vacina-x",
            "content": "Checagem comprova que alegação é falsa.",
            "score": 0.92,
            # published_date ausente propositalmente
        },
    ]
}


class TestTavilySearchAdapter(unittest.TestCase):
    """Testes do adaptador Tavily."""

    def _adapter_com_chave(self) -> TavilySearchAdapter:
        return TavilySearchAdapter(api_key="fake-key-for-test")

    # ---- Caso 1: Sucesso na busca ----
    @patch("search.requests.post")
    def test_busca_sucesso(self, mock_post: MagicMock):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _TAVILY_SUCESSO_RESPONSE
        mock_post.return_value = mock_resp

        adapter = self._adapter_com_chave()
        evidencias = adapter.buscar_evidencias_por_alegacao("vacina X causa infertilidade")

        self.assertEqual(len(evidencias), 2)
        self.assertIsInstance(evidencias[0], Evidencia)
        self.assertEqual(evidencias[0].title, "Vacina X é segura, diz OMS")
        self.assertEqual(evidencias[0].url, "https://who.int/news/vacina-x")
        self.assertEqual(mock_post.call_args.kwargs["json"]["max_results"], 8)

    # ---- Caso 2: Datas ausentes retornam None ----
    @patch("search.requests.post")
    def test_published_at_none_quando_ausente(self, mock_post: MagicMock):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _TAVILY_SUCESSO_RESPONSE
        mock_post.return_value = mock_resp

        adapter = self._adapter_com_chave()
        evidencias = adapter.buscar_evidencias_por_alegacao("vacina X")

        # Primeiro resultado tem data
        self.assertEqual(evidencias[0].published_at, "2025-03-15")
        # Segundo resultado não tem published_date
        self.assertIsNone(evidencias[1].published_at)

    # ---- Caso 3: Score preservado como relevância ----
    @patch("search.requests.post")
    def test_score_como_relevancia(self, mock_post: MagicMock):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _TAVILY_SUCESSO_RESPONSE
        mock_post.return_value = mock_resp

        adapter = self._adapter_com_chave()
        evidencias = adapter.buscar_evidencias_por_alegacao("vacina X")

        self.assertAlmostEqual(evidencias[0].relevance_score, 0.87)
        self.assertAlmostEqual(evidencias[1].relevance_score, 0.92)

    # ---- Caso 4: Classificação de source_type ----
    @patch("search.requests.post")
    def test_source_type_classificacao(self, mock_post: MagicMock):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _TAVILY_SUCESSO_RESPONSE
        mock_post.return_value = mock_resp

        adapter = self._adapter_com_chave()
        evidencias = adapter.buscar_evidencias_por_alegacao("vacina X")

        # who.int → secondary (domínio reconhecível, não oficial/fact-check)
        self.assertEqual(evidencias[0].source_type, "secondary")
        # aosfatos.org → fact_check
        self.assertEqual(evidencias[1].source_type, "fact_check")

    # ---- Caso 5: Erro/Timeout tratado graciosamente ----
    @patch("search.requests.post")
    def test_timeout_retorna_vazio(self, mock_post: MagicMock):
        requests_mod = sys.modules["requests"]
        mock_post.side_effect = requests_mod.exceptions.Timeout("timeout")

        adapter = self._adapter_com_chave()
        evidencias = adapter.buscar_evidencias_por_alegacao("qualquer alegação")

        self.assertEqual(evidencias, [])

    @patch("search.requests.post")
    def test_erro_500_retorna_vazio(self, mock_post: MagicMock):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_post.return_value = mock_resp

        adapter = self._adapter_com_chave()
        evidencias = adapter.buscar_evidencias_por_alegacao("qualquer alegação")

        self.assertEqual(evidencias, [])

    @patch("search.requests.post")
    def test_connection_error_retorna_vazio(self, mock_post: MagicMock):
        requests_mod = sys.modules["requests"]
        mock_post.side_effect = requests_mod.exceptions.ConnectionError("connection refused")

        adapter = self._adapter_com_chave()
        evidencias = adapter.buscar_evidencias_por_alegacao("qualquer alegação")

        self.assertEqual(evidencias, [])

    # ---- Caso 6: API key ausente ----
    def test_sem_api_key_retorna_vazio(self):
        adapter = TavilySearchAdapter(api_key=None)
        evidencias = adapter.buscar_evidencias_por_alegacao("qualquer alegação")
        self.assertEqual(evidencias, [])

    def test_sem_api_key_nao_crash(self):
        """Garante que a ausência de chave não gera exceção."""
        adapter = TavilySearchAdapter(api_key=None)
        try:
            adapter.buscar_evidencias_por_alegacao("alegação teste")
        except Exception as e:
            self.fail(f"Adapter com chave ausente levantou exceção: {e}")

    # ---- Verificações adicionais ----
    def test_alegacao_vazia_retorna_vazio(self):
        adapter = self._adapter_com_chave()
        self.assertEqual(adapter.buscar_evidencias_por_alegacao(""), [])
        self.assertEqual(adapter.buscar_evidencias_por_alegacao("   "), [])

    @patch("search.requests.post")
    def test_url_nunca_perdida(self, mock_post: MagicMock):
        """AC05: url nunca deve ser perdida."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _TAVILY_SUCESSO_RESPONSE
        mock_post.return_value = mock_resp

        adapter = self._adapter_com_chave()
        evidencias = adapter.buscar_evidencias_por_alegacao("vacina X")

        for ev in evidencias:
            self.assertTrue(ev.url, "URL não deve ser vazia")


class TestWebSearchProviderInterface(unittest.TestCase):
    """Verifica que os adapters implementam a interface corretamente (AC12)."""

    def test_tavily_eh_websearchprovider(self):
        adapter = TavilySearchAdapter(api_key=None)
        self.assertIsInstance(adapter, WebSearchProvider)

    def test_google_eh_websearchprovider(self):
        adapter = GoogleFactCheckAdapter(api_key=None)
        self.assertIsInstance(adapter, WebSearchProvider)


if __name__ == "__main__":
    unittest.main()
