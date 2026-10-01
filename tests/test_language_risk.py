"""Inferência e integração sem Telegram, chaves ou chamadas externas."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import test_search  # Instala os stubs antes de importar config (não lê .env).
from claims import Alegacao, ResultadoExtracao
from language_risk import AnalisadorLinguistico, ROOT
from motor_ia import MotorIA
from verifier import VerificadorAlegacoes

try:
    import joblib
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline
    HAS_ML = True
except ImportError:
    HAS_ML = False


class TestMissingModel(unittest.TestCase):
    def test_missing_file_does_not_require_ml(self):
        with patch("language_risk.Path.is_file", return_value=False):
            analyzer = AnalisadorLinguistico("models/missing.joblib")
        self.assertEqual(analyzer.model_path, ROOT / "models/missing.joblib")
        self.assertIsNone(analyzer.analisar("Texto"))
        self.assertIn("não encontrado", analyzer.indisponibilidade)


@unittest.skipUnless(HAS_ML, "Instale requirements-inference.txt para testar o pipeline real")
class TestLanguageRisk(unittest.TestCase):
    def setUp(self):
        # Artefato sintético exclusivo dos testes, não substitui a baseline real.
        self.folder = tempfile.TemporaryDirectory(dir=ROOT, prefix=".test-model-")
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "fixture.joblib"
        self.pipeline = Pipeline([
            ("features", FeatureUnion([
                ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))),
            ], n_jobs=1)),
            ("classifier", LogisticRegression(solver="liblinear", random_state=42)),
        ]).fit([
            "Relatório oficial divulga dados do estudo", "Universidade publica pesquisa científica",
            "URGENTE segredo milagroso compartilhe agora", "Bomba cura milagrosa que escondem",
        ], [0, 0, 1, 1])
        joblib.dump(self.pipeline, self.path)

    def test_serialized_pipeline_matches_training_and_loads_once(self):
        with patch("joblib.load", wraps=joblib.load) as load:
            analyzer = AnalisadorLinguistico(self.path)
            for text in ["Relatório oficial divulga dados", "URGENTE compartilhe agora"]:
                expected = self.pipeline.predict_proba([text])[0, 1]
                self.assertAlmostEqual(analyzer.analisar(text), expected)
                self.assertAlmostEqual(analyzer.analisar(text), expected)
            load.assert_called_once_with(self.path)
        self.assertEqual(analyzer._pipeline.get_params()["features__n_jobs"], 1)

    def test_invalid_or_unfitted_artifact(self):
        self.path.write_bytes(b"not a model")
        self.assertIsNone(AnalisadorLinguistico(self.path).analisar("Texto"))
        joblib.dump(Pipeline([("classifier", LogisticRegression())]), self.path)
        analyzer = AnalisadorLinguistico(self.path)
        self.assertIsNone(analyzer.analisar("Texto"))
        self.assertIn("inválido", analyzer.indisponibilidade)

    def test_missing_dependency_and_version_mismatch(self):
        with patch.dict(sys.modules, {"joblib": None}):
            analyzer = AnalisadorLinguistico(self.path)
        self.assertIsNone(analyzer.analisar("Texto"))
        self.assertIn("dependências", analyzer.indisponibilidade)
        import warnings
        from sklearn.exceptions import InconsistentVersionWarning

        def incompatible(_path):
            warnings.warn(InconsistentVersionWarning(
                estimator_name="LogisticRegression", current_sklearn_version="1.9.1",
                original_sklearn_version="0.0",
            ))

        with patch("joblib.load", side_effect=incompatible):
            analyzer = AnalisadorLinguistico(self.path)
        self.assertIsNone(analyzer.analisar("Texto"))
        self.assertIn("incompatível", analyzer.indisponibilidade)

    def test_reversed_class_order_and_invalid_probabilities(self):
        pipeline = MagicMock()
        pipeline.classes_ = np.array([1, 0])
        with patch("joblib.load", return_value=pipeline), patch("sklearn.utils.validation.check_is_fitted"):
            analyzer = AnalisadorLinguistico(self.path)
        pipeline.predict_proba.return_value = np.array([[0.8, 0.2]])
        self.assertEqual(analyzer.analisar("Texto"), 0.8)
        for probabilities in [[[float("nan"), 0.2]], [[1.2, -0.2]], [[0.3, 0.3]], [[0.8]]]:
            with self.subTest(probabilities=probabilities):
                pipeline.predict_proba.return_value = np.array(probabilities)
                self.assertIsNone(analyzer.analisar("Texto"))
        pipeline.predict_proba.side_effect = RuntimeError("inference failure")
        self.assertIsNone(analyzer.analisar("Texto"))

    def test_unexpected_labels_are_rejected(self):
        pipeline = MagicMock()
        pipeline.classes_ = np.array(["true", "fake"])
        with patch("joblib.load", return_value=pipeline), patch("sklearn.utils.validation.check_is_fitted"):
            analyzer = AnalisadorLinguistico(self.path)
        self.assertIsNone(analyzer.analisar("Texto"))
        pipeline.predict_proba.assert_not_called()


class TestMotorLanguageRisk(unittest.IsolatedAsyncioTestCase):
    async def test_original_text_score_and_factual_abstention_are_separate(self):
        text = "Notícia completa: o relatório foi divulgado. Opinião do autor."
        analyzer = MagicMock()
        analyzer.analisar.return_value = 0.93
        analyzer.indisponibilidade = None
        with patch("motor_ia.ExtratorAlegações") as extractor, patch("verifier.OPENAI_API_KEY", None):
            extractor.return_value.extrair.return_value = ResultadoExtracao(text, [Alegacao(1, "O relatório foi divulgado")])
            motor = MotorIA([], VerificadorAlegacoes(), analyzer)
        result = await motor.analisar_texto(text)
        analyzer.analisar.assert_called_once_with(text)
        self.assertEqual(result.score_linguistico, 0.93)
        self.assertIn("0,930", result.analise_linguistica)
        self.assertIn("não confirma", result.analise_linguistica)
        self.assertEqual(result.vereditos[0].categoria, "evidencia_insuficiente")
        self.assertFalse(result.requer_revisao)
        from bot import BotTelegram
        self.assertIn(result.analise_linguistica, BotTelegram._formatar_relatorio(result))

    async def test_unavailable_score_keeps_factual_processing(self):
        analyzer = MagicMock()
        analyzer.analisar.return_value = None
        analyzer.indisponibilidade = "modelo treinado não encontrado"
        with patch("motor_ia.ExtratorAlegações") as extractor, patch("verifier.OPENAI_API_KEY", None):
            extractor.return_value.extrair.return_value = ResultadoExtracao("Texto", [Alegacao(1, "Alegação")])
            motor = MotorIA([], VerificadorAlegacoes(), analyzer)
        result = await motor.analisar_texto("Texto")
        self.assertIsNone(result.score_linguistico)
        self.assertIn("indisponível", result.analise_linguistica)
        self.assertEqual(result.vereditos[0].categoria, "evidencia_insuficiente")


if __name__ == "__main__":
    unittest.main()
