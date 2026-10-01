"""Configuração sem LLM e fallback de extração, sem ler .env."""
import unittest
from unittest.mock import patch

import test_search
import config
from claims import ExtratorAlegações


class TestOptionalOpenAI(unittest.TestCase):
    def test_telegram_token_is_sufficient(self):
        with patch.object(config, "TELEGRAM_BOT_TOKEN", "test-token"), patch.object(config, "OPENAI_API_KEY", None):
            config.validate_env()

    def test_telegram_token_still_required(self):
        with patch.object(config, "TELEGRAM_BOT_TOKEN", None), patch.object(config, "OPENAI_API_KEY", None):
            with self.assertRaisesRegex(ValueError, "TELEGRAM_BOT_TOKEN") as error:
                config.validate_env()
        self.assertNotIn("OPENAI_API_KEY", str(error.exception))

    def test_rules_extract_without_creating_openai_client(self):
        with patch("claims.OPENAI_API_KEY", None), patch("claims.OpenAI") as client:
            extractor = ExtratorAlegações()
            result = extractor.extrair("A água ferve a 100 graus Celsius ao nível do mar.")
        client.assert_not_called()
        self.assertIsNone(extractor.client)
        self.assertEqual(len(result.alegacoes), 1)
        self.assertIn("100 graus Celsius", result.alegacoes[0].texto)
