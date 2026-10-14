"""Inferência local da baseline; o score não é uma verificação factual."""

import hashlib
import json
import logging
import math
import warnings
from pathlib import Path

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = ROOT / "models/language_risk/baseline_tfidf_logreg.joblib"
DEFAULT_MLP_METRICS_PATH = ROOT / "data/processed/language_risk/bertimbau/metrics_mlp.json"
DEFAULT_LOGREG_METRICS_PATH = ROOT / "data/processed/language_risk/bertimbau/metrics.json"
DEFAULT_BERTIMBAU_MODEL_DIR = ROOT / "models/language_risk/bertimbau_base"


def _absolute_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_model_path(artifact: dict) -> Path:
    model_path = _absolute_path(artifact["path"]).resolve()
    model_root = (ROOT / "models/language_risk").resolve()
    if not model_path.is_relative_to(model_root):
        raise ValueError("Checkpoint fora da pasta de modelos")
    if _sha256(model_path) != artifact["sha256"]:
        raise ValueError("Checkpoint diferente do registrado no treino")
    return model_path


def _load_bertimbau_encoder(encoder_dir: Path, config: dict, device: str):
    from bertimbau_embeddings import BertimbauEncoder

    if config["pooling"] != "mean of all content tokens across non-overlapping chunks":
        raise ValueError("Pooling do encoder incompatível")
    if _sha256(encoder_dir / "pytorch_model.bin") != config["checkpoint_sha256"]:
        raise ValueError("Pesos do encoder diferem dos usados no treino")
    encoder = BertimbauEncoder(
        encoder_dir, device=device, max_tokens=int(config["max_tokens"])
    )
    if encoder.dimension != int(config["dimension"]):
        raise ValueError("Dimensão do encoder incompatível")
    return encoder


class AnalisadorLinguistico:
    """Carrega uma vez o pipeline treinado, sem refazer fit ou alterar o texto."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH):
        self.model_path = _absolute_path(model_path)
        self._pipeline = None
        self._fake_column = None
        self.descricao_modelo = "TF-IDF + regressão logística"
        self.indisponibilidade = "modelo treinado não encontrado"
        if not self.model_path.is_file():
            logger.warning("Modelo linguístico não encontrado.")
            return
        try:
            import joblib
            from sklearn.exceptions import InconsistentVersionWarning
            from sklearn.utils.validation import check_is_fitted
            import warnings

            # Joblib deve receber somente o artefato de treino confiável do projeto.
            with warnings.catch_warnings():
                warnings.simplefilter("error", InconsistentVersionWarning)
                pipeline = joblib.load(self.model_path)
            check_is_fitted(pipeline.named_steps["classifier"])
            classes = list(pipeline.classes_)
            if len(classes) != 2 or set(classes) != {0, 1}:
                raise ValueError("O modelo deve usar 0=true e 1=fake")
            # Evita subprocessos por mensagem; não altera vocabulário ou pesos.
            pipeline.set_params(features__n_jobs=1)
            self._fake_column = classes.index(1)
            self._pipeline = pipeline
            self.indisponibilidade = None
            logger.info("Modelo linguístico carregado.")
        except ImportError:
            self.indisponibilidade = "dependências de inferência não instaladas"
            logger.warning("Dependências do modelo linguístico indisponíveis.")
        except Exception:
            self.indisponibilidade = "modelo incompatível ou inválido"
            logger.warning("Não foi possível carregar o modelo linguístico.")

    def analisar(self, texto: str) -> float | None:
        """Retorna P(classe 1 | texto) ou ausência de score em caso de falha."""
        if self._pipeline is None:
            return None
        try:
            probabilities = self._pipeline.predict_proba([texto])
            if probabilities.shape != (1, 2):
                raise ValueError("Dimensões de probabilidade inválidas")
            values = [float(value) for value in probabilities[0]]
            if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
                raise ValueError("Probabilidade inválida")
            if not math.isclose(sum(values), 1.0, abs_tol=1e-6):
                raise ValueError("Probabilidades não normalizadas")
            return values[self._fake_column]
        except Exception:
            logger.warning("Falha na inferência linguística; análise factual preservada.")
            return None


class AnalisadorBertimbauMLP:
    """Aplica BERTimbau congelado e o checkpoint escolhido na validação."""

    descricao_modelo = "BERTimbau Base + rede neural MLP"

    def __init__(
        self,
        metrics_path: str | Path = DEFAULT_MLP_METRICS_PATH,
        encoder_dir: str | Path = DEFAULT_BERTIMBAU_MODEL_DIR,
        device: str = "auto",
    ):
        self.metrics_path = _absolute_path(metrics_path)
        self.encoder_dir = _absolute_path(encoder_dir)
        self._encoder = None
        self._model = None
        self._mean = None
        self._scale = None
        self._threshold = None
        self._torch = None
        self.indisponibilidade = "artefatos da MLP não encontrados"
        if not self.metrics_path.is_file():
            logger.warning("Relatório de seleção da MLP não encontrado.")
            return
        try:
            import numpy as np
            import torch

            from bertimbau_mlp import build_model

            selection = json.loads(self.metrics_path.read_text(encoding="utf-8"))
            if selection.get("experiment") != "bertimbau_frozen_mlp":
                raise ValueError("Relatório de seleção inválido")
            seed = selection["best_validation_run"]["seed"]
            run = next(item for item in selection["runs"] if item["seed"] == seed)
            artifact = run["artifact"]
            model_path = _verified_model_path(artifact)
            encoder_config = selection["encoder"]
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
            dimension = int(encoder_config["dimension"])
            if checkpoint["seed"] != seed or checkpoint["dimension"] != dimension:
                raise ValueError("Dimensão ou semente do checkpoint inválida")
            model = build_model(
                dimension, checkpoint["hidden_size"], checkpoint["dropout"]
            )
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            mean = checkpoint["scaler_mean"].numpy()
            scale = checkpoint["scaler_scale"].numpy()
            if (
                mean.shape != (dimension,)
                or scale.shape != (dimension,)
                or not np.isfinite(mean).all()
                or not np.isfinite(scale).all()
                or not (scale > 0).all()
            ):
                raise ValueError("Normalização do checkpoint inválida")
            threshold = float(checkpoint["threshold"])
            if not math.isfinite(threshold) or not 0 <= threshold <= 1:
                raise ValueError("Limiar do checkpoint inválido")
            encoder = _load_bertimbau_encoder(
                self.encoder_dir, encoder_config, device
            )
            self._encoder = encoder
            self._model = model
            self._mean = mean
            self._scale = scale
            self._threshold = threshold
            self._torch = torch
            self.indisponibilidade = None
            logger.info("Modelo linguístico BERTimbau + MLP carregado.")
        except ImportError:
            self.indisponibilidade = "dependências BERTimbau não instaladas"
            logger.warning("Dependências do BERTimbau indisponíveis.")
        except Exception:
            self.indisponibilidade = "modelo BERTimbau ou MLP incompatível"
            logger.warning("Não foi possível carregar BERTimbau + MLP.")

    def analisar(self, texto: str) -> float | None:
        """Retorna P(classe fake | texto), sem modificar a verificação factual."""
        if self._model is None:
            return None
        try:
            import numpy as np

            vector = self._encoder.encode([texto])
            if vector.shape != (1, len(self._mean)) or not np.isfinite(vector).all():
                raise ValueError("Embedding inválido")
            normalized = (vector - self._mean) / self._scale
            with self._torch.inference_mode():
                logits = self._model(self._torch.from_numpy(normalized))
                score = float(self._torch.sigmoid(logits).item())
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Probabilidade inválida")
            return score
        except Exception:
            logger.warning("Falha na inferência BERTimbau; análise factual preservada.")
            return None


class AnalisadorBertimbauLogReg:
    """Aplica BERTimbau congelado e a regressão logística treinada nos vetores."""

    descricao_modelo = "BERTimbau Base + regressão logística"

    def __init__(
        self,
        metrics_path: str | Path = DEFAULT_LOGREG_METRICS_PATH,
        encoder_dir: str | Path = DEFAULT_BERTIMBAU_MODEL_DIR,
        device: str = "auto",
    ):
        self.metrics_path = _absolute_path(metrics_path)
        self.encoder_dir = _absolute_path(encoder_dir)
        self._encoder = None
        self._pipeline = None
        self._fake_column = None
        self.indisponibilidade = "artefatos da regressão BERTimbau não encontrados"
        if not self.metrics_path.is_file():
            logger.warning("Relatório da regressão BERTimbau não encontrado.")
            return
        try:
            import joblib
            from sklearn.exceptions import InconsistentVersionWarning
            from sklearn.utils.validation import check_is_fitted

            report = json.loads(self.metrics_path.read_text(encoding="utf-8"))
            if report.get("experiment") != "baseline_bertimbau_frozen_logistic_regression":
                raise ValueError("Relatório de regressão inválido")
            model_path = _verified_model_path(report["artifacts"]["model"])
            with warnings.catch_warnings():
                warnings.simplefilter("error", InconsistentVersionWarning)
                pipeline = joblib.load(model_path)
            check_is_fitted(pipeline.named_steps["classifier"])
            classes = list(pipeline.classes_)
            if len(classes) != 2 or set(classes) != {0, 1}:
                raise ValueError("O modelo deve usar 0=true e 1=fake")
            encoder = _load_bertimbau_encoder(
                self.encoder_dir, report["encoder"], device
            )
            if pipeline.named_steps["classifier"].n_features_in_ != encoder.dimension:
                raise ValueError("Classificador e encoder usam dimensões diferentes")
            self._encoder = encoder
            self._pipeline = pipeline
            self._fake_column = classes.index(1)
            self.indisponibilidade = None
            logger.info("Modelo linguístico BERTimbau + regressão carregado.")
        except ImportError:
            self.indisponibilidade = "dependências BERTimbau não instaladas"
            logger.warning("Dependências da regressão BERTimbau indisponíveis.")
        except Exception:
            self.indisponibilidade = "modelo BERTimbau ou regressão incompatível"
            logger.warning("Não foi possível carregar BERTimbau + regressão.")

    def analisar(self, texto: str) -> float | None:
        if self._pipeline is None:
            return None
        try:
            import numpy as np

            vector = self._encoder.encode([texto])
            if vector.shape != (1, self._encoder.dimension) or not np.isfinite(vector).all():
                raise ValueError("Embedding inválido")
            probabilities = self._pipeline.predict_proba(vector)
            if probabilities.shape != (1, 2):
                raise ValueError("Dimensões de probabilidade inválidas")
            values = [float(value) for value in probabilities[0]]
            if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
                raise ValueError("Probabilidade inválida")
            if not math.isclose(sum(values), 1.0, abs_tol=1e-6):
                raise ValueError("Probabilidades não normalizadas")
            return values[self._fake_column]
        except Exception:
            logger.warning("Falha na inferência BERTimbau; análise factual preservada.")
            return None


def formatar_analise_linguistica(
    score: float | None,
    motivo: str | None = None,
    modelo: str = "TF-IDF + regressão logística",
) -> str:
    if score is None:
        return f"Análise linguística indisponível: {motivo or 'falha na inferência'}."
    value = f"{score:.3f}".replace(".", ",")
    return (
        f"Score experimental de associação à classe fake: {value} (escala de 0 a 1).\n"
        f"Modelo local: {modelo}.\n"
        "Esse score reflete padrões aprendidos nas bases de treino; não confirma "
        "se a notícia é falsa e não substitui a verificação das alegações."
    )
