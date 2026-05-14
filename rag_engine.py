"""
RAG Engine — fast vector similarity search.

Indexing: embeds text chunks locally (no LLM calls, completes in seconds).
Querying: finds relevant chunks via cosine similarity, then asks Ollama to answer.

Chunks and embeddings are saved to disk so uploads survive app restarts.
"""
import os
import json
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

WORKING_DIR = os.getenv("LIGHTRAG_DIR", "./lightrag_storage")

# LLM backend — Groq if key is set, otherwise local Ollama
_GROQ_KEY  = os.getenv("GROQ_API_KEY")
LLM_BASE   = "https://api.groq.com/openai/v1" if _GROQ_KEY else "http://localhost:11434/v1"
LLM_KEY    = _GROQ_KEY or "ollama"
LLM_MODEL  = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile") if _GROQ_KEY else os.getenv("OLLAMA_MODEL", "llama3.2")


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _load_embed_model():
    from sentence_transformers import SentenceTransformer
    logger.info("Loading local embedding model...")
    m = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Embedding model ready.")
    return m


def _llm_chat(messages: list[dict], max_tokens: int = 2048) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=LLM_KEY, base_url=LLM_BASE)
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.1,
    )
    return resp.choices[0].message.content


def _cosine_top_k(embeddings: np.ndarray, query_emb: np.ndarray, k: int) -> list[int]:
    q = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    e = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    sims = (e @ q).flatten()
    k = min(k, len(sims))
    return np.argsort(sims)[-k:][::-1].tolist()


# ------------------------------------------------------------------ #
# Engine                                                               #
# ------------------------------------------------------------------ #

class RAGEngine:
    def __init__(self):
        self._chunks: list[str]      = []
        self._doc_names: list[str]   = []
        self._embeddings: np.ndarray | None = None
        self._indexed_docs: list[str] = []
        self._embed_model = None
        self._init()

    def _init(self):
        try:
            Path(WORKING_DIR).mkdir(parents=True, exist_ok=True)
            self._embed_model = _load_embed_model()
            self._load_from_disk()
            logger.info(f"RAG engine ready — {len(self._chunks)} chunks, {len(self._indexed_docs)} docs.")
        except Exception as e:
            logger.error(f"RAG init failed: {e}")

    # ---- persistence ---- #

    def _save_to_disk(self):
        try:
            meta = {"chunks": self._chunks, "doc_names": self._doc_names, "indexed_docs": self._indexed_docs}
            Path(f"{WORKING_DIR}/chunks.json").write_text(json.dumps(meta))
            if self._embeddings is not None:
                np.save(f"{WORKING_DIR}/embeddings.npy", self._embeddings)
        except Exception as e:
            logger.warning(f"Save failed: {e}")

    def _load_from_disk(self):
        try:
            cf = Path(f"{WORKING_DIR}/chunks.json")
            ef = Path(f"{WORKING_DIR}/embeddings.npy")
            if cf.exists() and ef.exists():
                meta = json.loads(cf.read_text())
                self._chunks       = meta["chunks"]
                self._doc_names    = meta["doc_names"]
                self._indexed_docs = meta["indexed_docs"]
                self._embeddings   = np.load(str(ef))
                logger.info(f"Loaded {len(self._chunks)} chunks from disk.")
        except Exception as e:
            logger.warning(f"Load failed: {e}")

    # ---- indexing ---- #

    def insert_document(self, text: str, doc_name: str = "document") -> bool:
        if not self._embed_model:
            logger.error("Embedding model not available.")
            return False
        try:
            from pdf_processor import chunk_text
            chunks = chunk_text(text, chunk_size=800, overlap=80)
            if not chunks:
                return False

            logger.info(f"Embedding {len(chunks)} chunks for '{doc_name}'...")
            new_emb = self._embed_model.encode(
                chunks, show_progress_bar=False, convert_to_numpy=True
            )
            self._chunks.extend(chunks)
            self._doc_names.extend([doc_name] * len(chunks))
            self._embeddings = (
                new_emb if self._embeddings is None
                else np.vstack([self._embeddings, new_emb])
            )
            self._indexed_docs.append(doc_name)
            self._save_to_disk()
            logger.info(f"Indexed '{doc_name}': {len(chunks)} chunks.")
            return True
        except Exception as e:
            logger.error(f"insert_document error for '{doc_name}': {e}")
            return False

    # ---- retrieval ---- #

    def _retrieve_context(self, query: str, top_k: int = 5) -> str:
        if self._embeddings is None or not self._chunks:
            return ""
        q_emb = self._embed_model.encode([query], convert_to_numpy=True)[0]
        idxs  = _cosine_top_k(self._embeddings, q_emb, top_k)
        return "\n\n---\n\n".join(self._chunks[i] for i in idxs)

    def query(self, question: str, only_need_context: bool = False) -> str:
        if not self._indexed_docs:
            return "⚠️ No documents indexed. Please upload a PDF first."
        if not self._embed_model:
            return "❌ Embedding model not available."

        context = self._retrieve_context(question, top_k=5)
        if not context:
            return "No relevant information found."

        if only_need_context:
            return context

        try:
            return _llm_chat([
                {"role": "system", "content": "Answer using only the provided context. Be concise and accurate."},
                {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ], max_tokens=1024)
        except Exception as e:
            logger.error(f"Query LLM error: {e}")
            return f"❌ LLM error: {e}"

    def query_stream(self, question: str):
        """Stream the LLM response token by token for instant feedback."""
        if not self._indexed_docs:
            yield "⚠️ No documents indexed. Please upload a PDF first."
            return
        if not self._embed_model:
            yield "❌ Embedding model not available."
            return

        context = self._retrieve_context(question, top_k=5)
        if not context:
            yield "No relevant information found."
            return

        try:
            from openai import OpenAI
            client = OpenAI(api_key=LLM_KEY, base_url=LLM_BASE)
            with client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "Answer using only the provided context. Be concise and accurate."},
                    {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}"},
                ],
                max_tokens=1024,
                temperature=0.1,
                stream=True,
            ) as stream:
                for chunk in stream:
                    token = chunk.choices[0].delta.content
                    if token:
                        yield token
        except Exception as e:
            logger.error(f"query_stream error: {e}")
            yield f"❌ LLM error: {e}"

    def get_context(self, topic: str) -> str:
        return self._retrieve_context(topic, top_k=6)

    # ---- properties ---- #

    @property
    def has_documents(self) -> bool:
        return bool(self._indexed_docs)

    @property
    def document_count(self) -> int:
        return len(self._indexed_docs)

    def list_documents(self) -> list[str]:
        return list(self._indexed_docs)
