"""Contratos da LLM local sem iniciar Ollama durante o CI."""

from datetime import datetime, timedelta
import unittest

import test_search
from claims import ExtratorAlegações
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


class TestExtracaoQwen(unittest.TestCase):
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
            {"pergunta": "Qual fonte primária poderia confirmar essa informação?"},
            {"pergunta": "Que detalhe da fonte ainda deixa espaço para dúvida?"},
            {"pergunta": "Que contexto você buscaria antes de compartilhar?"},
        ])
        dialogo = GerenciadorSocratico(llm=llm)
        inicio = dialogo.iniciar_sessao(7, "Notícia", self.vereditos, chat_id=11)
        segunda = dialogo.processar_resposta(7, "Eu procuraria a prefeitura", chat_id=11)
        terceira = dialogo.processar_resposta(7, "A data não está clara", chat_id=11)
        sintese = dialogo.processar_resposta(7, "Eu aguardaria", chat_id=11)
        self.assertIn("Pergunta 1/3", inicio)
        self.assertIn('Você mencionou "Eu procuraria a prefeitura"', segunda)
        self.assertIn("Pergunta 3/3", terceira)
        self.assertIn("evidência insuficiente", sintese)
        self.assertIn("reflexões (não tratadas como fatos)", sintese)
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
        self.assertIn("fonte ou evidência", inicio)

    def test_question_that_denies_existing_sources_uses_safe_fallback(self):
        llm = FakeLLM([{
            "pergunta": "Como a ausência de fontes afeta a análise desta alegação?"
        }])
        dialogo = GerenciadorSocratico(llm=llm)
        inicio = dialogo.iniciar_sessao(7, "Notícia", self.vereditos, chat_id=11)
        self.assertIn("Que fonte ou evidência", inicio)
        self.assertNotIn("ausência de fontes", inicio)
