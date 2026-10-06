"""Inferência local da baseline; o score não é uma verificação factual."""

import logging
import math
from pathlib import Path

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = ROOT / "models/language_risk/baseline_tfidf_logreg.joblib"


class AnalisadorLinguistico:
    """Carrega uma vez o pipeline treinado, sem refazer fit ou alterar o texto."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH):
        path = Path(model_path).expanduser()
        self.model_path = path if path.is_absolute() else ROOT / path
        self._pipeline = None
        self._fake_column = None
        self.indisponibilidade = "modelo treinado não encontrado"
        if not self.model_path.is_file():
            logger.warning("Modelo linguístico não encontrado: %s", self.model_path)
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
            logger.info("Modelo linguístico carregado: %s", self.model_path)
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


def formatar_analise_linguistica(score: float | None, motivo: str | None = None) -> str:
    if score is None:
        return f"Análise linguística indisponível: {motivo or 'falha na inferência'}."
    value = f"{score:.3f}".replace(".", ",")
    return (
        f"Score experimental de associação à classe fake: {value} (escala de 0 a 1).\n"
        "Modelo local: TF-IDF + regressão logística.\n"
        "Esse score reflete padrões aprendidos nas bases de treino; não confirma "
        "se a notícia é falsa e não substitui a verificação das alegações."
    )
