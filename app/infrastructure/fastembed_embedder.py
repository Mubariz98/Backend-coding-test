"""FastEmbed implementation of the `Embedder` port.

FastEmbed runs ONNX models locally on CPU (no API key, no PyTorch).
The model is downloaded once into `cache_dir` on first use.
"""

from fastembed import TextEmbedding

from app.shared.ports import Vector


class FastEmbedEmbedder:
    def __init__(self, model_name: str, cache_dir: str):
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        self.dimension: int = self._model.embedding_size

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        return [v.tolist() for v in self._model.embed(texts)]

    def embed_query(self, text: str) -> Vector:
        return self.embed_documents([text])[0]
