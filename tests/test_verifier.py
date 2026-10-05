"""Regressões MVP-10: nenhuma chamada externa ou leitura de .env."""
from copy import deepcopy
import json
import sys
import types
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

# Reutiliza os stubs de dependências da suíte existente antes de importar config.
import test_search
from claims import Alegacao, ResultadoExtracao
from motor_ia import MotorIA
from search import Evidencia, GoogleFactCheckAdapter, TavilySearchAdapter
from verifier import VerificadorAlegacoes, VerificadorAlegações, _id_evidencia

CLAIM = "A vacina X não causa infertilidade"


def evidence(text=CLAIM, rating=None, url="https://fonte.example/checagem", checked=CLAIM):
    return Evidencia(
        title="Título original", url=url, publisher="Fonte",
        snippet=f"Veredito do Checador: {rating}" if rating else text,
        source_type="fact_check" if rating else "news",
        checked_claim=checked if rating else None, textual_rating=rating,
    )


def payload(evs, relations, category="apoiada"):
    return {
        "categoria": category, "justificativa": "Comparação com os registros fornecidos.",
        "referencias": [
            {"id": _id_evidencia(ev), "trecho": ev.snippet, "relacao": rel}
            for ev, rel in zip(evs, relations)
        ],
    }


def response(data, finish="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish,
        message=SimpleNamespace(content=data if isinstance(data, str) else json.dumps(data)),
    )])


class TestVerifier(unittest.TestCase):
    def setUp(self):
        self.key_patch = patch("verifier.OPENAI_API_KEY", None)
        self.key_patch.start()
        self.addCleanup(self.key_patch.stop)
        self.claim = Alegacao(1, CLAIM)

    def verify_model(self, evs, data, finish="stop"):
        client = MagicMock()
        client.chat.completions.create.return_value = response(data, finish)
        verifier = VerificadorAlegacoes(client=client)
        return verifier.verificar_alegacao(self.claim, evs), client

    def test_import_alias_and_default_providers(self):
        self.assertIs(VerificadorAlegações, VerificadorAlegacoes)
        with patch("motor_ia.ExtratorAlegações"):
            motor = MotorIA()
        self.assertIsInstance(motor.provedores_busca[0], GoogleFactCheckAdapter)
        self.assertIsInstance(motor.provedores_busca[1], TavilySearchAdapter)

    def test_no_evidence_no_llm(self):
        client = MagicMock()
        result = VerificadorAlegacoes(client).verificar_alegacao(self.claim, [])
        self.assertEqual(result.categoria, "evidencia_insuficiente")
        self.assertEqual(result.fontes_citadas, [])
        client.chat.completions.create.assert_not_called()

    def test_exact_ratings_not_substrings(self):
        for rating, expected in [
            ("Incorreto", "refutada"), ("Correto", "apoiada"),
            ("Não comprovado", "nao_verificada"),
            ("O fato continua sob investigação", "nao_verificada"),
            ("Não é verdadeiro", "nao_verificada"),
        ]:
            with self.subTest(rating=rating):
                ev = evidence(rating=rating)
                result = VerificadorAlegacoes().verificar_alegacao(self.claim, [ev])
                self.assertEqual(result.categoria, expected)

    def test_web_keywords_without_comparison_abstain(self):
        for text in ["Incorreto", "Não comprovado", "O fato continua sob investigação"]:
            with self.subTest(text=text):
                result = VerificadorAlegacoes().verificar_alegacao(self.claim, [evidence(text)])
                self.assertEqual(result.categoria, "nao_verificada")

    def test_factcheck_other_claim_missing_text_and_negation(self):
        client = MagicMock()
        for checked in [None, "A vacina X causa infertilidade", "A vacina Y não causa infertilidade"]:
            with self.subTest(checked=checked):
                result = VerificadorAlegacoes(client).verificar_alegacao(
                    self.claim, [evidence(rating="Falso", checked=checked)],
                )
                self.assertEqual(result.categoria, "nao_verificada")
                self.assertEqual(result.fontes_citadas, [])
        client.chat.completions.create.assert_not_called()

    def test_conflicting_factchecks_cannot_be_binary(self):
        evs = [evidence(rating="Falso"), evidence(rating="Verdadeiro", url="https://outra.example/c")]
        client = MagicMock()
        verifier = VerificadorAlegacoes(client)
        result = verifier.verificar_alegacao(self.claim, evs)
        self.assertEqual(result.categoria, "conflitante")
        self.assertEqual(set(result.fontes_citadas), {ev.url for ev in evs})
        for ev in evs:
            self.assertIn(ev.snippet, result.justificativa)
        client.chat.completions.create.assert_not_called()
        self.assertEqual(result, verifier.verificar_alegacao(self.claim, evs[::-1] + evs))

    def test_model_conflict_and_negation(self):
        evs = [evidence(), evidence("A vacina X causa infertilidade", url="https://outra.example/c")]
        result, _ = self.verify_model(evs, payload(evs, ["apoia", "refuta"], "conflitante"))
        self.assertEqual(result.categoria, "conflitante")
        invalid, _ = self.verify_model(evs, payload(evs, ["apoia", "refuta"], "apoiada"))
        self.assertEqual(invalid.categoria, "nao_verificada")
        result, _ = self.verify_model([evs[1]], payload([evs[1]], ["refuta"], "refutada"))
        self.assertEqual(result.categoria, "refutada")

    def test_invalid_model_contracts_abstain(self):
        evs = [evidence()]
        valid = payload(evs, ["apoia"])
        invalids = [None, [], "{", "```json\n{}\n```", {"categoria": "apoiada"}]
        for field, value in [
            ("categoria", "verdadeira"), ("categoria", []),
            ("justificativa", ""), ("justificativa", "  "), ("justificativa", 42),
            ("referencias", "url"), ("referencias", []),
        ]:
            item = deepcopy(valid)
            item[field] = value
            invalids.append(item)
        for field, value in [
            ("id", "https://inventada.example/"), ("id", []),
            ("trecho", "Uma afirmação inventada"), ("trecho", 1),
            ("relacao", "falso"), ("relacao", []),
        ]:
            item = deepcopy(valid)
            item["referencias"][0][field] = value
            invalids.append(item)
        invented_url = deepcopy(valid)
        invented_url["fontes_citadas"] = ["https://inventada.example/"]
        invalids.append(invented_url)
        extra_url = deepcopy(valid)
        extra_url["referencias"][0]["url"] = "https://inventada.example/"
        invalids.append(extra_url)
        duplicate = deepcopy(valid)
        duplicate["referencias"] *= 2
        invalids.append(duplicate)
        for data in invalids:
            with self.subTest(data=data):
                result, _ = self.verify_model(evs, data)
                self.assertEqual(result.categoria, "nao_verificada")
                self.assertEqual(result.fontes_citadas, [])
                self.assertNotIn("apoiada", result.justificativa)

    def test_truncated_response_and_provider_failure(self):
        evs = [evidence()]
        result, _ = self.verify_model(evs, payload(evs, ["apoia"]), "length")
        self.assertEqual(result.categoria, "nao_verificada")
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("offline")
        result = VerificadorAlegacoes(client).verificar_alegacao(self.claim, evs)
        self.assertEqual(result.categoria, "nao_verificada")

    def test_model_cannot_override_unknown_factcheck_rating(self):
        evs = [evidence(rating="Não comprovado")]
        result, _ = self.verify_model(evs, payload(evs, ["apoia"]))
        self.assertEqual(result.categoria, "nao_verificada")

    def test_no_omitted_evidence(self):
        evs = [evidence(), evidence("Refutação", url="https://outra.example/c")]
        result, _ = self.verify_model(evs, payload(evs[:1], ["apoia"]))
        self.assertEqual(result.categoria, "nao_verificada")

    def test_stable_context_and_repeated_results(self):
        evs = [evidence(), evidence("Outra fonte", url="https://outra.example/c")]
        client = MagicMock()
        client.chat.completions.create.return_value = response(payload(evs, ["apoia", "insuficiente"]))
        verifier = VerificadorAlegacoes(client)
        result = verifier.verificar_alegacao(self.claim, evs)
        self.assertEqual(result.categoria, "apoiada")
        self.assertEqual(result, verifier.verificar_alegacao(self.claim, evs[::-1] + evs))
        client.chat.completions.create.assert_called_once()
        context = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        second_client = MagicMock()
        second_client.chat.completions.create.return_value = response(payload(evs[::-1], ["insuficiente", "apoia"]))
        other = VerificadorAlegacoes(second_client).verificar_alegacao(self.claim, evs[::-1])
        self.assertEqual(result, other)
        self.assertEqual(context, second_client.chat.completions.create.call_args.kwargs["messages"][1]["content"])
        result.fontes_citadas.clear()
        self.assertTrue(verifier.verificar_alegacao(self.claim, evs).fontes_citadas)

    def test_only_validated_data_in_rationale(self):
        evs = [evidence()]
        data = payload(evs, ["apoia"])
        data["justificativa"] = "Segundo https://inventada.example/ houve uma descoberta inventada."
        result, _ = self.verify_model(evs, data)
        self.assertNotIn("inventada", result.justificativa)
        self.assertIn(evs[0].snippet, result.justificativa)
        self.assertEqual(result.fontes_citadas, [evs[0].url])


class TestGoogleMetadata(unittest.TestCase):
    @patch("search.requests.get")
    def test_metadata_and_original_claim_not_fabricated(self, get):
        get.return_value.status_code = 200
        get.return_value.json.return_value = {"claims": [
            {"text": "Outra alegação", "claimDate": "2025-01-01", "claimReview": [{
                "publisher": {"name": "Checador"}, "title": "Título da checagem",
                "url": "https://checador.example/c", "textualRating": "Falso",
                "reviewDate": "2025-02-01",
            }]},
            {"claimDate": "2024-01-01", "claimReview": [{"url": "https://checador.example/d"}]},
        ]}
        evs = GoogleFactCheckAdapter(api_key="fake-key").buscar_evidencias_por_alegacao(CLAIM)
        self.assertEqual(evs[0].checked_claim, "Outra alegação")
        self.assertEqual(evs[0].title, "Título da checagem")
        self.assertEqual(evs[0].publisher, "Checador")
        self.assertEqual(evs[0].textual_rating, "Falso")
        self.assertEqual(evs[0].snippet, "Veredito do Checador: Falso")
        self.assertEqual(evs[0].published_at, "2025-02-01")
        self.assertEqual(evs[0].source_type, "fact_check")
        self.assertIsNone(evs[1].checked_claim)
        self.assertIsNone(evs[1].textual_rating)
        self.assertEqual(evs[1].published_at, "2024-01-01")
        self.assertNotIn(CLAIM, evs[1].title)


class TestMotor(unittest.IsolatedAsyncioTestCase):
    async def test_separate_claims_and_failed_provider(self):
        claims = [Alegacao(1, CLAIM), Alegacao(2, "A medida Y entrou em vigor")]
        evs = [evidence(rating="Falso"), evidence(rating="Verdadeiro", checked=claims[1].texto, url="https://outra.example/c")]
        failing = MagicMock()
        failing.buscar_evidencias_por_alegacao.side_effect = RuntimeError("offline")
        working = MagicMock()
        working.buscar_evidencias_por_alegacao.side_effect = [[evs[0]], [evs[1]]]
        with patch("motor_ia.ExtratorAlegações") as extractor, patch("verifier.OPENAI_API_KEY", None):
            extractor.return_value.extrair.return_value = ResultadoExtracao("texto", claims)
            motor = MotorIA([failing, working], VerificadorAlegacoes())
        result = await motor.analisar_texto("texto")
        self.assertEqual([v.categoria for v in result.vereditos], ["refutada", "apoiada"])
        for index, verdict in enumerate(result.vereditos):
            self.assertEqual(verdict.alegacao_id, claims[index].id)
            self.assertEqual(verdict.texto_alegacao, claims[index].texto)
            self.assertEqual(verdict.fontes_citadas, [evs[index].url])
            self.assertNotIn(evs[1-index].url, verdict.justificativa)
            self.assertIn(verdict.justificativa, result.resultado_factual)
        self.assertEqual(working.buscar_evidencias_por_alegacao.call_count, 2)

        # Importa o consumidor real com apenas a dependência Telegram simulada.
        telegram = types.ModuleType("telegram")
        ext = types.ModuleType("telegram.ext")
        for name in ["InlineKeyboardButton", "InlineKeyboardMarkup", "Update"]:
            setattr(telegram, name, MagicMock())
        for name in ["ApplicationBuilder", "CommandHandler", "ContextTypes", "MessageHandler", "filters"]:
            setattr(ext, name, MagicMock())
        with patch.dict(sys.modules, {"telegram": telegram, "telegram.ext": ext}):
            from bot import BotTelegram
            report = BotTelegram._formatar_relatorio(result)
        for verdict in result.vereditos:
            self.assertIn(verdict.texto_alegacao, report)
            self.assertIn(verdict.categoria, report)
            evidence_id, evidence_text = verdict.justificativa.split("] ", 1)
            self.assertIn(evidence_text, report)
            self.assertNotIn(evidence_id.lstrip("["), report)
        self.assertIn("Fonte indisponível", report)

    async def test_empty_provider_list_and_empty_input(self):
        with patch("motor_ia.ExtratorAlegações") as extractor, patch("verifier.OPENAI_API_KEY", None):
            extractor.return_value.extrair.return_value = ResultadoExtracao("texto", [Alegacao(1, CLAIM)])
            motor = MotorIA([])
        result = await motor.analisar_texto("texto")
        self.assertEqual(result.vereditos[0].categoria, "evidencia_insuficiente")
        self.assertIn("Conclusão: evidência insuficiente", result.resultado_factual)
        with self.assertRaises(ValueError):
            await motor.analisar_texto(" ")


if __name__ == "__main__":
    unittest.main()
