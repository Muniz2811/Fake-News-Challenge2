"""Inferência e integração sem Telegram, chaves ou chamadas externas."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import test_search  # noqa: F401  # Instala os stubs antes de importar config.
from claims import Alegacao, ResultadoExtracao
from language_risk import (
    AnalisadorBertimbauLogReg,
    AnalisadorBertimbauMLP,
    AnalisadorLinguistico,
    ROOT,
    formatar_analise_linguistica,
)
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


@unittest.skipUnless(HAS_ML, "Instale requirements.txt para testar o pipeline real")
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


@unittest.skipUnless(
    HAS_ML and importlib.util.find_spec("torch"),
    "Instale requirements/ml.txt para testar a MLP",
)
class TestBertimbauMLPRuntime(unittest.TestCase):
    def setUp(self):
        model_root = ROOT / "models/language_risk"
        model_root.mkdir(parents=True, exist_ok=True)
        self.folder = tempfile.TemporaryDirectory(
            dir=model_root, prefix="test-bertimbau-runtime-"
        )
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name)
        self.encoder_dir = self.path / "encoder"
        self.encoder_dir.mkdir()
        weights = self.encoder_dir / "pytorch_model.bin"
        weights.write_bytes(b"pesos sinteticos")
        self.encoder_hash = hashlib.sha256(weights.read_bytes()).hexdigest()
        import torch

        from bertimbau_mlp import build_model

        model = build_model(3, 4, 0.2)
        self.checkpoint = self.path / "mlp.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "scaler_mean": torch.zeros(3),
                "scaler_scale": torch.ones(3),
                "dimension": 3,
                "hidden_size": 4,
                "dropout": 0.2,
                "threshold": 0.5,
                "seed": 43,
            },
            self.checkpoint,
        )
        self.report = self.path / "metrics_mlp.json"
        self.report.write_text(
            json.dumps(
                {
                    "experiment": "bertimbau_frozen_mlp",
                    "best_validation_run": {"seed": 43},
                    "runs": [
                        {
                            "seed": 43,
                            "artifact": {
                                "path": str(self.checkpoint.relative_to(ROOT)),
                                "sha256": hashlib.sha256(
                                    self.checkpoint.read_bytes()
                                ).hexdigest(),
                            },
                        }
                    ],
                    "encoder": {
                        "pooling": "mean of all content tokens across non-overlapping chunks",
                        "checkpoint_sha256": self.encoder_hash,
                        "dimension": 3,
                        "max_tokens": 256,
                    },
                }
            ),
            encoding="utf-8",
        )

    def test_loaded_mlp_scores_original_text_and_names_model(self):
        encoder = MagicMock()
        encoder.dimension = 3
        encoder.encode.return_value = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
        with patch("bertimbau_embeddings.BertimbauEncoder", return_value=encoder):
            analyzer = AnalisadorBertimbauMLP(self.report, self.encoder_dir)
        self.assertIsNone(analyzer.indisponibilidade)
        score = analyzer.analisar("Notícia original")
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)
        encoder.encode.assert_called_once_with(["Notícia original"])
        self.assertIn("BERTimbau Base + rede neural MLP", formatar_analise_linguistica(
            score, modelo=analyzer.descricao_modelo
        ))

    def test_bad_artifact_abstains_without_loading_encoder(self):
        self.checkpoint.write_bytes(b"artefato alterado")
        with patch("bertimbau_embeddings.BertimbauEncoder") as encoder:
            analyzer = AnalisadorBertimbauMLP(self.report, self.encoder_dir)
        encoder.assert_not_called()
        self.assertIsNone(analyzer.analisar("Notícia"))
        self.assertIn("incompatível", analyzer.indisponibilidade)

    def test_bertimbau_logreg_uses_validated_encoder_and_classifier(self):
        from sklearn.preprocessing import StandardScaler

        pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(solver="liblinear")),
        ]).fit(
            np.asarray([[-2, 0, 0], [-1, 0, 0], [1, 0, 0], [2, 0, 0]]),
            [0, 0, 1, 1],
        )
        model_path = self.path / "logreg.joblib"
        joblib.dump(pipeline, model_path)
        metrics_path = self.path / "metrics_logreg.json"
        metrics_path.write_text(
            json.dumps({
                "experiment": "baseline_bertimbau_frozen_logistic_regression",
                "artifacts": {"model": {
                    "path": str(model_path.relative_to(ROOT)),
                    "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                }},
                "encoder": {
                    "pooling": "mean of all content tokens across non-overlapping chunks",
                    "checkpoint_sha256": self.encoder_hash,
                    "dimension": 3,
                    "max_tokens": 256,
                },
            }),
            encoding="utf-8",
        )
        encoder = MagicMock()
        encoder.dimension = 3
        encoder.encode.return_value = np.asarray([[1.5, 0, 0]], dtype=np.float32)
        with patch("bertimbau_embeddings.BertimbauEncoder", return_value=encoder):
            analyzer = AnalisadorBertimbauLogReg(metrics_path, self.encoder_dir)
        self.assertIsNone(analyzer.indisponibilidade)
        self.assertAlmostEqual(
            analyzer.analisar("Notícia original"),
            pipeline.predict_proba([[1.5, 0, 0]])[0, 1],
        )
        encoder.encode.assert_called_once_with(["Notícia original"])
        self.assertIn("BERTimbau Base + regressão logística", formatar_analise_linguistica(
            analyzer.analisar("Notícia original"), modelo=analyzer.descricao_modelo
        ))


class TestMotorLanguageRisk(unittest.IsolatedAsyncioTestCase):
    async def test_bertimbau_logreg_backend_is_selected(self):
        analyzer = MagicMock()
        analyzer.analisar.return_value = 0.4
        analyzer.indisponibilidade = None
        analyzer.descricao_modelo = "BERTimbau Base + regressão logística"
        with patch("motor_ia.LANGUAGE_RISK_BACKEND", "bertimbau_logreg"), patch(
            "motor_ia.AnalisadorBertimbauLogReg", return_value=analyzer
        ) as constructor, patch("motor_ia.ExtratorAlegações") as extractor:
            extractor.return_value.extrair.return_value = ResultadoExtracao(
                "Notícia", []
            )
            motor = MotorIA([], VerificadorAlegacoes())
        result = await motor.analisar_texto("Notícia")
        constructor.assert_called_once()
        self.assertIn("BERTimbau Base + regressão logística", result.analise_linguistica)
        self.assertEqual(result.score_linguistico, 0.4)

    async def test_mlp_backend_is_selected_and_named_in_report(self):
        analyzer = MagicMock()
        analyzer.analisar.return_value = 0.7
        analyzer.indisponibilidade = None
        analyzer.descricao_modelo = "BERTimbau Base + rede neural MLP"
        with patch("motor_ia.LANGUAGE_RISK_BACKEND", "bertimbau_mlp"), patch(
            "motor_ia.AnalisadorBertimbauMLP", return_value=analyzer
        ) as constructor, patch("motor_ia.ExtratorAlegações") as extractor:
            extractor.return_value.extrair.return_value = ResultadoExtracao(
                "Notícia", []
            )
            motor = MotorIA([], VerificadorAlegacoes())
        result = await motor.analisar_texto("Notícia")
        constructor.assert_called_once()
        self.assertIn("BERTimbau Base + rede neural MLP", result.analise_linguistica)
        self.assertEqual(result.score_linguistico, 0.7)

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
        # MVP-13: abstenção factual exige revisão, independentemente do score.
        self.assertTrue(result.requer_revisao)
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

    async def test_unavailable_provider_is_reported_once(self):
        analyzer = MagicMock()
        analyzer.analisar.return_value = None
        analyzer.indisponibilidade = "modelo treinado não encontrado"
        provider = MagicMock()
        provider.buscar_evidencias_por_alegacao.side_effect = RuntimeError("timeout")
        with patch("motor_ia.ExtratorAlegações") as extractor, patch("verifier.OPENAI_API_KEY", None):
            extractor.return_value.extrair.return_value = ResultadoExtracao(
                "Texto", [Alegacao(1, "Alegação A"), Alegacao(2, "Alegação B")],
            )
            motor = MotorIA([provider], VerificadorAlegacoes(), analyzer)
        result = await motor.analisar_texto("Texto")
        self.assertEqual(
            result.etapas_indisponiveis, ["Fonte indisponível: a busca no MagicMock não respondeu."],
        )
        self.assertEqual(len(result.vereditos), 2)


if __name__ == "__main__":
    unittest.main()
