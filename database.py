"""
Database — Supabase integration for document metadata and chat history.
Uses pgvector for embedding storage.

Setup SQL (run once in Supabase SQL editor):
--------------------------------------------
create extension if not exists vector;

create table if not exists documents (
    id          uuid primary key default gen_random_uuid(),
    filename    text not null,
    title       text,
    pages       int,
    chunk_index int,
    content     text not null,
    embedding   vector(1536),
    created_at  timestamptz default now()
);

create index on documents using ivfflat (embedding vector_cosine_ops)
    with (lists = 100);

create table if not exists chat_sessions (
    id         uuid primary key default gen_random_uuid(),
    session_id text not null,
    role       text not null,   -- 'user' | 'assistant'
    content    text not null,
    created_at timestamptz default now()
);
--------------------------------------------
"""
import os
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class Database:
    def __init__(self):
        self.client = None
        self._init_client()

    def _init_client(self):
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            logger.warning("SUPABASE_URL or SUPABASE_KEY not set — database features disabled.")
            return
        try:
            from supabase import create_client
            self.client = create_client(url, key)
            logger.info("Supabase connected.")
        except ImportError:
            logger.warning("supabase-py not installed — database features disabled.")
        except Exception as e:
            logger.warning(f"Supabase connection failed: {e}")

    @property
    def available(self) -> bool:
        return self.client is not None

    # ------------------------------------------------------------------ #
    # Document storage                                                     #
    # ------------------------------------------------------------------ #

    def store_chunk(
        self,
        filename: str,
        title: str,
        pages: int,
        chunk_index: int,
        content: str,
        embedding: Optional[list[float]] = None,
    ) -> Optional[str]:
        """Insert one document chunk into Supabase."""
        if not self.available:
            return None
        try:
            row = {
                "filename": filename,
                "title": title,
                "pages": pages,
                "chunk_index": chunk_index,
                "content": content,
            }
            if embedding:
                row["embedding"] = embedding
            result = self.client.table("documents").insert(row).execute()
            return result.data[0]["id"] if result.data else None
        except Exception as e:
            logger.error(f"store_chunk error: {e}")
            return None

    def list_documents(self) -> list[str]:
        """Return distinct filenames indexed in Supabase."""
        if not self.available:
            return []
        try:
            result = (
                self.client.table("documents")
                .select("filename")
                .execute()
            )
            seen = set()
            names = []
            for row in result.data or []:
                fn = row["filename"]
                if fn not in seen:
                    seen.add(fn)
                    names.append(fn)
            return names
        except Exception as e:
            logger.error(f"list_documents error: {e}")
            return []

    # ------------------------------------------------------------------ #
    # Chat history                                                         #
    # ------------------------------------------------------------------ #

    def save_message(self, session_id: str, role: str, content: str):
        if not self.available:
            return
        try:
            self.client.table("chat_sessions").insert(
                {"session_id": session_id, "role": role, "content": content}
            ).execute()
        except Exception as e:
            logger.error(f"save_message error: {e}")

    def get_history(self, session_id: str, limit: int = 20) -> list[dict]:
        if not self.available:
            return []
        try:
            result = (
                self.client.table("chat_sessions")
                .select("role,content,created_at")
                .eq("session_id", session_id)
                .order("created_at")
                .limit(limit)
                .execute()
            )
            return result.data or []
        except Exception as e:
            logger.error(f"get_history error: {e}")
            return []
