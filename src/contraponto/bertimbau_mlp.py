"""Arquitetura compartilhada pelo treino e pela inferência da MLP."""


def build_model(dimension: int, hidden_size: int, dropout: float):
    """Constrói a rede binária que recebe embeddings BERTimbau congelados."""
    from torch import nn

    return nn.Sequential(
        nn.Linear(dimension, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, 1),
    )
