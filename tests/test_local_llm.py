"""Contratos da LLM local sem iniciar Ollama durante o CI."""

from datetime import datetime, timedelta
import unittest
from unittest.mock import patch

import test_search  # noqa: F401  # Instala stubs antes dos módulos da aplicação.
from claims import ExtratorAlegações
from llm import _validar_base_url, criar_provedor_llm
from socratic import GerenciadorSocratico
from verifier import VereditoAlegacao


class FakeLLM:
    def __init__(self, respostas):
        self.respostas = list(respostas)
        self.chamadas = []

    def gerar_json(self, sistema, entrada, schema, nome):
        self.chamadas.append((sistema, entrada, schema, nome))
        resposta = self.respostas.pop(0)
        if isinstance(resposta, Exception):
            raise resposta
        return resposta


class TestTransporteLLM(unittest.TestCase):
    def test_https_and_local_ollama_are_allowed(self):
        for url in (
            "https://llm.example/v1",
            "http://localhost:11434/v1",
            "http://127.0.0.1:11434/v1",
            "http://host.docker.internal:11434/v1",
        ):
            with self.subTest(url=url):
                _validar_base_url(url)

    def test_external_plain_http_is_rejected(self):
        for url in ("http://llm.example/v1", "ftp://llm.example/v1", "https://"):
            with self.subTest(url=url):
                with self.assertRaisesRegex(ValueError, "HTTPS"):
                    _validar_base_url(url)

    def test_tailscale_host_header_is_scoped_and_keeps_the_original_https_url(self):
        for base_url, expected in (
            ("https://desktop.example.ts.net/v1", {"Host": "localhost"}),
            ("https://DESKTOP.example.TS.NET/v1", {"Host": "localhost"}),
            ("http://localhost:11434/v1", {}),
            ("https://llm.example/v1", {}),
            ("https://example.ts.net.evil.example/v1", {}),
        ):
            with self.subTest(base_url=base_url), patch("llm.OpenAI") as factory, patch.multiple(
                "llm", LLM_BASE_URL=base_url, LLM_API_KEY="ollama", LLM_MODEL="qwen3:8b",
                OPENAI_API_KEY=None,
            ):
                provider = criar_provedor_llm()
                options = factory.call_args.kwargs
                self.assertEqual(options["base_url"], base_url)
                self.assertEqual(options["default_headers"], expected)
                self.assertEqual(options["max_retries"], 0)
                self.assertNotIn("http_client", options)  # Não desabilita a validação TLS.
                self.assertEqual(provider.model, "qwen3:8b")


class TestExtracaoQwen(unittest.TestCase):
    def test_preserves_thousands_separators_and_decimal_in_valid_llm_origin(self):
        texto = (
            "Segundo o IBGE, o Brasil tinha 203.080.756 habitantes em 1º de agosto de 2022. "
            "A densidade demográfica era de 23,86 habitantes por quilômetro quadrado."
        )
        primeira, segunda = texto.split(". A densidade")
        primeira += "."
        segunda = "A densidade" + segunda
        llm = FakeLLM([{"alegacoes": [
            {"texto": primeira, "trecho_origem": primeira},
            {"texto": segunda, "trecho_origem": segunda},
        ]}])
        resultado = ExtratorAlegações(llm=llm).extrair(texto)
        self.assertEqual([item.texto for item in resultado.alegacoes], [primeira, segunda])
        self.assertEqual(resultado.alegacoes[1].posicao, texto.index("A densidade"))

    def test_fallback_keeps_ibge_numbers_in_complete_claims(self):
        texto = (
            "O Censo 2022 registrou 203.080.756 habitantes no Brasil. "
            "Desse total, 177.508.417 viviam em áreas urbanas e 25.572.339 "
            "em áreas rurais. A densidade foi de 23,86 habitantes por km²."
        )
        resultado = ExtratorAlegações(llm=FakeLLM([ValueError("resposta inválida")])).extrair(texto)
        self.assertEqual(len(resultado.alegacoes), 3)
        self.assertIn("203.080.756 habitantes", resultado.alegacoes[0].texto)
        self.assertIn("177.508.417", resultado.alegacoes[1].texto)
        self.assertIn("25.572.339", resultado.alegacoes[1].texto)
        self.assertIn("23,86", resultado.alegacoes[2].texto)

    def test_validates_origin_and_assigns_id_and_position_in_python(self):
        texto = "Introdução. A prefeitura abrirá 20 escolas em dezembro."
        llm = FakeLLM([{"alegacoes": [{
            "texto": "A prefeitura abrirá 20 escolas em dezembro.",
            "trecho_origem": "A prefeitura abrirá 20 escolas em dezembro.",
        }]}])
        resultado = ExtratorAlegações(limit=3, llm=llm).extrair(texto)
        self.assertEqual(resultado.alegacoes[0].id, 1)
        self.assertEqual(resultado.alegacoes[0].posicao, texto.index("A prefeitura"))
        self.assertEqual(llm.chamadas[0][3], "extracao_alegacoes")

    def test_rejects_the_overbroad_origin_returned_in_manual_test(self):
        texto = (
            "A prefeitura informou que 20 escolas serão inauguradas em dezembro. "
            "Na minha opinião, isso é maravilhoso."
        )
        llm = FakeLLM([{"alegacoes": [{
            "texto": "A prefeitura informou que 20 escolas serão inauguradas em dezembro.",
            "trecho_origem": texto,
        }]}])
        resultado = ExtratorAlegações(llm=llm).extrair(texto)
        self.assertEqual(len(resultado.alegacoes), 1)
        self.assertEqual(
            resultado.alegacoes[0].trecho_origem,
            "A prefeitura informou que 20 escolas serão inauguradas em dezembro",
        )
        self.assertNotIn("opinião", resultado.alegacoes[0].trecho_origem)

    def test_invalid_output_or_outage_uses_rules(self):
        texto = "A vacina não foi proibida em 2026."
        for resposta in [RuntimeError("offline"), {"alegacoes": "inválido"}]:
            with self.subTest(resposta=resposta):
                resultado = ExtratorAlegações(llm=FakeLLM([resposta])).extrair(texto)
                self.assertEqual(len(resultado.alegacoes), 1)
                self.assertIn("não", resultado.alegacoes[0].texto)
                self.assertIn("2026", resultado.alegacoes[0].texto)


class TestDialogoSocratico(unittest.TestCase):
    def setUp(self):
        self.vereditos = [VereditoAlegacao(
            1,
            "A prefeitura abrirá 20 escolas.",
            "evidencia_insuficiente",
            "A fonte não informa a data.",
            ["https://fonte.example/noticia"],
        )]

    def test_three_questions_adaptation_and_validated_summary(self):
        llm = FakeLLM([
            {
                "relacao": "nao_comprovada_pela_investigacao",
                "ids_alegacoes_usadas": [1],
                "resposta": "A investigação não confirma que houve intenção política.",
                "pergunta": "O que fez você relacionar essa notícia a uma intenção política?",
            },
            {
                "relacao": "nao_comprovada_pela_investigacao",
                "ids_alegacoes_usadas": [1],
                "resposta": "As fontes não identificam filiação partidária das pessoas envolvidas.",
                "pergunta": "Que informação mostraria se essa suspeita tem fundamento?",
            },
            {
                "relacao": "nao_comprovada_pela_investigacao",
                "ids_alegacoes_usadas": [1],
                "resposta": "A manipulação mencionada não foi confirmada pelo material consultado.",
                "pergunta": "",
            },
        ])
        dialogo = GerenciadorSocratico(llm=llm)
        inicio = dialogo.iniciar_sessao(7, "Notícia", self.vereditos, chat_id=11)
        segunda = dialogo.processar_resposta(7, "Isso parece uma ação política", chat_id=11)
        terceira = dialogo.processar_resposta(7, "As pessoas eram de um partido", chat_id=11)
        sintese = dialogo.processar_resposta(7, "Então o resultado foi manipulado", chat_id=11)
        self.assertIn("Pergunta 1/3", inicio)
        self.assertIn("não confirma que houve intenção política", segunda)
        self.assertIn("relacionar essa notícia", segunda)
        self.assertNotIn("próxima afirmação", segunda)
        self.assertIn("Pergunta 3/3", terceira)
        self.assertIn("manipulação mencionada não foi confirmada", sintese)
        self.assertIn("permanecem como hipóteses", sintese)
        self.assertFalse(dialogo.tem_sessao_ativa(7, 11))
        self.assertEqual(len(llm.chamadas), 3)

    def test_sessions_are_isolated_cancelable_and_expire(self):
        dialogo = GerenciadorSocratico(llm=FakeLLM([]), ttl_minutos=30)
        dialogo.iniciar_sessao(7, "Notícia A", self.vereditos, chat_id=11)
        dialogo.iniciar_sessao(7, "Notícia B", self.vereditos, chat_id=12)
        self.assertIn("encerrado", dialogo.cancelar_sessao(7, 11))
        self.assertTrue(dialogo.tem_sessao_ativa(7, 12))
        chave = dialogo._chave(7, 12)
        dialogo._sessoes[chave].expira_em = datetime.now() - timedelta(seconds=1)
        self.assertFalse(dialogo.tem_sessao_ativa(7, 12))

    def test_llm_failure_keeps_dialogue_available(self):
        dialogo = GerenciadorSocratico(llm=FakeLLM([RuntimeError("offline")]))
        inicio = dialogo.iniciar_sessao(7, "Notícia", self.vereditos, chat_id=11)
        turno = dialogo.processar_resposta(7, "Isso foi manipulado", chat_id=11)
        self.assertIn("o que mais chamou sua atenção", inicio)
        self.assertIn("não aparece confirmada", turno)
        self.assertIn("O que levou você", turno)

    def test_invalid_or_technical_turn_uses_safe_fallback(self):
        llm = FakeLLM([{
            "relacao": "apoiada_pela_investigacao",
            "ids_alegacoes_usadas": [999],
            "resposta": "O usuário está certo.",
            "pergunta": "Qual score e metodologia confirmam sua hipótese?",
        }])
        dialogo = GerenciadorSocratico(llm=llm)
        dialogo.iniciar_sessao(7, "Notícia", self.vereditos, chat_id=11)
        turno = dialogo.processar_resposta(7, "Isso foi manipulado", chat_id=11)
        self.assertIn("não aparece confirmada", turno)
        self.assertNotIn("score", turno)

    def test_user_question_is_answered_before_next_reflection(self):
        llm = FakeLLM([
            {
                "relacao": "nao_comprovada_pela_investigacao",
                "ids_alegacoes_usadas": [1],
                "resposta": "A fonte consultada não informa a data da abertura.",
                "pergunta": "Essa ausência de data muda sua avaliação da notícia?",
            },
        ])
        dialogo = GerenciadorSocratico(llm=llm)
        dialogo.iniciar_sessao(7, "Notícia", self.vereditos, chat_id=11)
        turno = dialogo.processar_resposta(
            7, "Mas por que isso ainda não foi confirmado?", chat_id=11,
        )
        self.assertIn("não informa a data", turno)
        self.assertIn("Pergunta 2/3", turno)
        self.assertIn("muda sua avaliação", turno)

    def test_unconfirmed_new_claim_may_have_no_supporting_claim_id(self):
        llm = FakeLLM([{
            "relacao": "nao_comprovada_pela_investigacao",
            "ids_alegacoes_usadas": [],
            "resposta": "A investigação não menciona participação partidária na fiscalização.",
            "pergunta": "O que levou você a atribuir uma filiação partidária aos fiscais?",
        }])
        dialogo = GerenciadorSocratico(llm=llm)
        dialogo.iniciar_sessao(7, "Notícia", self.vereditos, chat_id=11)
        turno = dialogo.processar_resposta(
            7, "Os fiscais eram de um partido", chat_id=11,
        )
        self.assertIn("não menciona participação partidária", turno)
        self.assertIn("atribuir uma filiação", turno)

    def test_out_of_scope_is_redirected_but_disagreement_is_not(self):
        llm = FakeLLM([
            {
                "relacao": "fora_do_escopo",
                "ids_alegacoes_usadas": [],
                "resposta": "Resposta qualquer.",
                "pergunta": "Qual ponto da notícia você gostaria de retomar?",
            },
            {
                "relacao": "nao_comprovada_pela_investigacao",
                "ids_alegacoes_usadas": [1],
                "resposta": "A investigação não confirma que houve fraude.",
                "pergunta": "O que na notícia levou você a suspeitar de fraude?",
            },
        ])
        dialogo = GerenciadorSocratico(llm=llm)
        dialogo.iniciar_sessao(7, "Notícia", self.vereditos, chat_id=11)
        fora = dialogo.processar_resposta(7, "Qual seu filme favorito?", chat_id=11)
        relacionado = dialogo.processar_resposta(7, "Ainda acho que houve fraude", chat_id=11)
        self.assertIn("fugir do assunto", fora)
        self.assertIn("não confirma que houve fraude", relacionado)
        contexto = llm.chamadas[1][1]
        self.assertIn("Ainda acho que houve fraude", contexto)
        self.assertIn("A prefeitura abrirá 20 escolas", contexto)

    def test_political_suspicion_is_not_discarded_when_model_marks_out_of_scope(self):
        llm = FakeLLM([{
            "relacao": "fora_do_escopo",
            "ids_alegacoes_usadas": [],
            "resposta": "Isso não tem relação.",
            "pergunta": "Como você sabe que houve manipulação partidária?",
        }])
        dialogo = GerenciadorSocratico(llm=llm)
        dialogo.iniciar_sessao(7, "Notícia sobre eleições", self.vereditos, chat_id=11)
        turno = dialogo.processar_resposta(
            7, "Acho que a votação foi manipulada por petistas", chat_id=11,
        )
        self.assertIn("não confirma essa conclusão", turno)
        self.assertNotIn("fugir do assunto", turno)
        self.assertIn("não oferecem uma base segura", turno)
        self.assertIn("fazê-lo reconsiderá-la", turno)
