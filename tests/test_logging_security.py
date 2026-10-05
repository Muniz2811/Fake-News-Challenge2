"""Credenciais fictícias não devem aparecer nem em tracebacks de bibliotecas."""
import logging
import unittest
from unittest.mock import patch

import test_search  # Isola configuração; não lê .env.
import bot


class TestLoggingSecurity(unittest.TestCase):
    def test_redacts_formatted_arguments_and_exception_traceback(self):
        key = "tavily-fake-test-secret"
        try:
            raise RuntimeError(f"Request failed: https://api.example.org/?key={key}")
        except RuntimeError:
            import sys
            exception = sys.exc_info()
        record = logging.LogRecord(
            "provider", logging.ERROR, __file__, 1,
            "Failure at %s", ("https://api.telegram.org/bot123456:fake-test-token/getMe",),
            exception,
        )
        with patch.object(bot, "TAVILY_API_KEY", key):
            output = bot._RedactingFormatter("%(message)s").format(record)
        self.assertNotIn(key, output)
        self.assertNotIn("fake-test-token", output)
        self.assertIn("RuntimeError", output)
        self.assertIn("getMe", output)
        self.assertIn("CREDENCIAL OCULTA", output)
