"""Catálogo de modelos locais, sem dependência do Telegram ou de chaves externas."""
import json
from pathlib import Path
from threading import RLock
from typing import Callable

MODEL_NAMES = {
    "tfidf": "TF-IDF + regressão logística",
    "bertimbau_logreg": "BERTimbau + regressão logística",
    "bertimbau_mlp": "BERTimbau + MLP",
}


class ModelCatalog:
    """Carrega sob demanda e compartilha um encoder por configuração validada."""

    def __init__(self, factories: dict[str, Callable], initial: dict | None = None):
        self._factories = factories
        self._analyzers = dict(initial or {})
        self._encoders = {}
        self._lock = RLock()

    def get(self, model_id: str):
        if model_id not in MODEL_NAMES or model_id not in self._factories:
            raise ValueError("Modelo linguístico desconhecido.")
        if model_id in self._analyzers:
            return self._analyzers[model_id]
        with self._lock:
            if model_id not in self._analyzers:
                analyzer = self._factories[model_id]()
                # Permite tentar novamente após corrigir artefatos ausentes.
                if analyzer.indisponibilidade is None:
                    self._analyzers[model_id] = analyzer
                return analyzer
            return self._analyzers[model_id]

    def encoder(self, directory: Path, config: dict, device: str):
        from language_risk import _load_bertimbau_encoder

        key = (str(directory.resolve()), json.dumps(config, sort_keys=True), device)
        with self._lock:
            if key not in self._encoders:
                self._encoders[key] = _load_bertimbau_encoder(directory, config, device)
            return self._encoders[key]
