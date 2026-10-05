"""Fixture sintética de inferência Docker, sem bases reais ou configuração externa."""
import base64
import errno
import io
import json
import math
import os
import sys
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

TEXT = "Universidade publica pesquisa sobre qualidade da água"


def fixture():
    pipeline = Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2))),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))),
        ], n_jobs=1)),
        ("classifier", LogisticRegression(solver="liblinear", random_state=42)),
    ]).fit([
        "Relatório oficial divulga dados do estudo", "Universidade publica pesquisa científica",
        "URGENTE segredo milagroso compartilhe agora", "Bomba cura milagrosa que escondem",
    ], [0, 0, 1, 1])
    buffer = io.BytesIO()
    joblib.dump(pipeline, buffer)
    print(json.dumps({
        "artifact": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "expected": float(pipeline.predict_proba([TEXT])[0, 1]),
    }))


def verify(expected):
    from language_risk import AnalisadorLinguistico, formatar_analise_linguistica

    path = Path(os.environ["LANGUAGE_RISK_MODEL_PATH"])
    analyzer = AnalisadorLinguistico(path)
    assert analyzer.indisponibilidade is None, analyzer.indisponibilidade
    score = analyzer.analisar(TEXT)
    assert score is not None and math.isclose(score, expected, abs_tol=1e-12)
    assert f"{score:.3f}".replace(".", ",") in formatar_analise_linguistica(score)
    try:
        with (path.parent / "write-probe.txt").open("w"):
            pass
    except OSError as error:
        assert error.errno == errno.EROFS, "Falha deve ocorrer por montagem somente leitura"
    else:
        raise AssertionError("O modelo deve estar montado somente para leitura")
    print("Docker: modelo montado, score equivalente e pasta somente para leitura: OK")


if __name__ == "__main__":
    if sys.argv[1] == "fixture":
        fixture()
    elif sys.argv[1] == "verify":
        verify(float(sys.argv[2]))
    else:
        raise ValueError("Modo inválido")
