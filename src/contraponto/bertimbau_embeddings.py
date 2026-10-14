"""Embeddings de notícias com BERTimbau congelado, sem classificação factual."""

from pathlib import Path
from threading import Lock

import numpy as np


class BertimbauEncoder:
    """Faz média dos tokens de todos os blocos de uma notícia, sem truncá-la."""

    def __init__(
        self,
        model_dir: str | Path,
        device: str = "auto",
        max_tokens: int = 256,
        batch_size: int = 16,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self._inference_lock = Lock()
        self.model_dir = Path(model_dir)
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"Checkpoint não encontrado: {self.model_dir}")
        if batch_size < 1 or max_tokens < 4:
            raise ValueError("batch_size deve ser positivo e max_tokens >= 4")
        if device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            local_files_only=True,
            use_fast=True,
        )
        self.model = (
            AutoModel.from_pretrained(
                self.model_dir,
                local_files_only=True,
            )
            .to(self.device)
            .eval()
        )
        self.dimension = int(self.model.config.hidden_size)
        self.max_tokens = min(
            max_tokens, int(self.model.config.max_position_embeddings)
        )
        self.content_tokens = (
            self.max_tokens - self.tokenizer.num_special_tokens_to_add(False)
        )
        if self.content_tokens < 1:
            raise ValueError("max_tokens não deixa espaço para o conteúdo")

    def encode(self, texts: list[str]) -> np.ndarray:
        """Retorna um vetor float32 por notícia na mesma ordem da entrada."""
        # Os dois classificadores e vários usuários podem usar o mesmo encoder.
        with self._inference_lock:
            return self._encode(texts)

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        chunks: list[list[int]] = []
        owners: list[int] = []
        content_masks: list[list[int]] = []
        for owner, text in enumerate(texts):
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            if not token_ids:
                raise ValueError("Notícia sem tokens após a tokenização")
            for start in range(0, len(token_ids), self.content_tokens):
                content = token_ids[start : start + self.content_tokens]
                input_ids = self.tokenizer.build_inputs_with_special_tokens(content)
                if len(input_ids) != len(content) + 2:
                    raise ValueError(
                        "Esperados os marcadores [CLS] e [SEP] do BERTimbau"
                    )
                chunks.append(input_ids)
                content_masks.append([0] + [1] * len(content) + [0])
                owners.append(owner)

        sums = np.zeros((len(texts), self.dimension), dtype=np.float64)
        counts = np.zeros(len(texts), dtype=np.int64)
        for start in range(0, len(chunks), self.batch_size):
            end = min(start + self.batch_size, len(chunks))
            batch = self.tokenizer.pad(
                [{"input_ids": chunk} for chunk in chunks[start:end]],
                padding=True,
                return_tensors="pt",
            )
            mask = self.torch.zeros_like(batch["input_ids"], dtype=self.torch.float32)
            for row, values in enumerate(content_masks[start:end]):
                mask[row, : len(values)] = self.torch.tensor(
                    values, dtype=self.torch.float32
                )
            batch = {name: value.to(self.device) for name, value in batch.items()}
            with self.torch.inference_mode():
                hidden = self.model(**batch).last_hidden_state
                token_sums = (hidden * mask.to(self.device).unsqueeze(-1)).sum(dim=1)
            token_sums = token_sums.cpu().numpy()
            for row, owner in enumerate(owners[start:end]):
                sums[owner] += token_sums[row]
                counts[owner] += sum(content_masks[start + row])
        return (sums / counts[:, None]).astype(np.float32)
