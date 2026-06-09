"""SentenceTransformer wrapper around ``BAAI/bge-large-en-v1.5``.

The model (~1.3 GB, 1024-dim output) is lazily loaded into a thread-safe
process-wide singleton and auto-placed on the best available device:
``cuda`` > ``mps`` > ``cpu``.

BGE models recommend prefixing *queries* with an instruction string but
*not* documents, so this module exposes :func:`embed_query` and
:func:`embed_documents` separately to make that distinction explicit.
"""

import threading

from sentence_transformers import SentenceTransformer

from backend.utils.logger import get_logger

logger = get_logger(__name__)

EMBEDDING_MODEL_NAME: str = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIM: int = 1024
BGE_QUERY_INSTRUCTION: str = "Represent this sentence for searching relevant passages: "

_model: SentenceTransformer | None = None
_lock: threading.Lock = threading.Lock()


def _detect_device() -> str:
    """Return the best PyTorch device available: ``cuda`` > ``mps`` > ``cpu``."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def get_embedder() -> SentenceTransformer:
    """Lazy-load the singleton BGE-large embedder.

    First call downloads weights (cached in ``~/.cache/huggingface``) and may
    take 10–30 seconds. Subsequent calls return the cached instance.
    """
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is None:
            device = _detect_device()
            logger.info("loading_embedding_model", model=EMBEDDING_MODEL_NAME, device=device)
            _model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
            logger.info("embedding_model_loaded", dim=EMBEDDING_DIM, device=device)
    return _model


def embed_query(query: str) -> list[float]:
    """Embed a single retrieval query (with BGE's recommended instruction prefix)."""
    model = get_embedder()
    text = BGE_QUERY_INSTRUCTION + query
    vector = model.encode(
        text,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vector.tolist()


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed a batch of documents (no instruction prefix). Vectors are L2-normalized."""
    if not texts:
        return []
    model = get_embedder()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=32,
    )
    return vectors.tolist()
