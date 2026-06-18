"""
EmbeddingClient — pluggable embedding interface.

Two backends, controlled by EMBEDDING_PROVIDER in .env:
  local  → sentence-transformers (free, ~3-5s cold load)
  openai → OpenAI text-embedding-3-small (no load time, ~$0.005/full run, cached)

Regardless of backend, embeddings are cached to disk after the first run so
the cost (time or money) is paid exactly once per candidate set.
"""
from __future__ import annotations
import hashlib
import os
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from loguru import logger


class EmbeddingClient(ABC):
    """Abstract interface every embedding backend must satisfy."""

    @abstractmethod
    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        """Return (N, dim) float32 array of embeddings."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str: ...


# ── Local: sentence-transformers ──────────────────────────────────────────────

class LocalEmbedder(EmbeddingClient):
    """
    Wraps sentence-transformers SentenceTransformer.
    Model is lazy-loaded on first call; a warmup() call at startup
    moves the load off the first request.
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        self._model_name = model
        self._model = None

    def warmup(self):
        """Pre-load the model. Call this at server startup."""
        _ = self._get_model()

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"[LocalEmbedder] Loading model: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)
            logger.info(f"[LocalEmbedder] Model ready")
        return self._model

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        model = self._get_model()
        return model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            batch_size=32,
        ).astype(np.float32)

    @property
    def model_name(self) -> str:
        return self._model_name


# ── API: OpenAI-compatible embedding endpoint ─────────────────────────────────

class APIEmbedder(EmbeddingClient):
    """
    Calls an OpenAI-compatible embedding endpoint.
    Default: text-embedding-3-small (~$0.02/1M tokens, ~$0.005 for 500 profiles).
    Also works with any OpenAI-compatible provider (Together, Azure, Ollama, etc.).
    No model loading — cold start is instant.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        base_url: Optional[str] = None,
        batch_size: int = 64,
    ):
        self._model_name = model
        self._batch_size = batch_size
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._base_url = base_url  # None = OpenAI default

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        from openai import OpenAI
        client = OpenAI(api_key=self._api_key, base_url=self._base_url)

        all_embeddings: list[list[float]] = []

        # Batch to stay within token limits
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            logger.debug(f"[APIEmbedder] Embedding batch {i//self._batch_size + 1} ({len(batch)} texts)")
            response = client.embeddings.create(input=batch, model=self._model_name)
            # Sort by index to guarantee order
            sorted_data = sorted(response.data, key=lambda x: x.index)
            all_embeddings.extend(item.embedding for item in sorted_data)

        arr = np.array(all_embeddings, dtype=np.float32)
        if normalize:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = arr / np.maximum(norms, 1e-10)
        return arr

    @property
    def model_name(self) -> str:
        return self._model_name


# ── Factory ───────────────────────────────────────────────────────────────────

_embedder: Optional[EmbeddingClient] = None


def get_embedder() -> EmbeddingClient:
    """Return the configured embedding client (singleton)."""
    global _embedder
    if _embedder is not None:
        return _embedder

    from app.config import settings

    provider = getattr(settings, "embedding_provider", "local")

    if provider == "openai":
        _embedder = APIEmbedder(
            api_key=getattr(settings, "openai_api_key", None),
            model=getattr(settings, "openai_embedding_model", "text-embedding-3-small"),
            base_url=getattr(settings, "openai_base_url", None),
        )
        logger.info(f"[Embedder] Using API backend: {_embedder.model_name}")
    else:
        _embedder = LocalEmbedder(model=settings.embedding_model)
        logger.info(f"[Embedder] Using local backend: {_embedder.model_name}")

    return _embedder


def warmup_embedder():
    """Warm up the embedder at startup (only meaningful for local backend)."""
    emb = get_embedder()
    if isinstance(emb, LocalEmbedder):
        emb.warmup()
