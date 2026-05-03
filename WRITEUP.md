# AI Handbook Generator — Project Write-up

## Overview

The AI Handbook Generator is a full-stack AI application that allows users to upload PDF documents and generate comprehensive, 20,000-word structured handbooks through a conversational interface. It combines retrieval-augmented generation (RAG), a large language model, and a modern web UI to transform raw document collections into professional long-form content.

---

## Problem Statement

Professionals and students regularly accumulate large collections of PDFs — research papers, reports, lecture notes, and reference materials. Synthesising these into a coherent, structured document is time-consuming and mentally demanding. This tool automates that process: upload your sources, describe what you want, and receive a publication-quality handbook in minutes.

---

## Technology Stack

| Layer | Technology | Role |
|-------|-----------|------|
| **UI** | Gradio 6 | Web interface with tabs, file upload, streaming chat |
| **LLM** | Groq — Llama 3.3 70B | Fast cloud inference for handbook writing and Q&A |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) | Local text embeddings, no API cost |
| **RAG** | Custom vector similarity search | Retrieves relevant document chunks for context |
| **PDF parsing** | pdfplumber + pypdf | Text extraction with fallback support |
| **Database** | Supabase (PostgreSQL + pgvector) | Stores document chunks and chat history |
| **Runtime** | Python 3.10, asyncio | Async-capable backend |

---

## System Architecture

```
User uploads PDF
       │
       ▼
 pdf_processor.py          Extract text, split into chunks
       │
       ▼
  rag_engine.py            Embed chunks with all-MiniLM-L6-v2 (local)
       │                   Save embeddings to disk (numpy)
       ▼
  Supabase                 Store chunk metadata (optional)

User asks question / requests handbook
       │
       ▼
  rag_engine.py            Embed query → cosine similarity → top-k chunks
       │
       ▼
handbook_generator.py      Pass context + prompt to Groq Llama 3.3 70B
       │
       ▼
  Gradio UI                Stream response tokens to browser in real time
```

---

## Key Features

### 1. Fast PDF Indexing
Unlike systems that require LLM calls during document ingestion, this app uses a **local embedding model only** during upload. A 1MB PDF indexes in under 10 seconds. Embeddings are persisted to disk so documents survive app restarts.

### 2. Streaming Chat
Responses are streamed token-by-token from the Groq API to the browser, providing instant feedback rather than a blank wait screen. This is achieved by using the OpenAI-compatible streaming API and yielding tokens through Gradio's generator support.

### 3. LongWriter Handbook Generation
The handbook generation follows the **LongWriter / AgentWrite** technique from academic research:

- **Phase 1 — Plan**: The LLM generates a 12–16 section table of contents with target word counts per section (totalling ≥20,000 words)
- **Phase 2 — Write**: Each section is written individually (~1,500 words each), with the relevant RAG context and a rolling summary of previous sections passed as input
- **Phase 3 — Assemble**: All sections are joined into a single Markdown document and made available for download

This approach bypasses the context window output limit — instead of asking the LLM to write 20,000 words in one call (impossible), it makes ~14 focused calls of ~1,500 words each.

### 4. Hybrid Storage
The app functions fully without Supabase configured. When Supabase credentials are provided, it additionally persists document chunk metadata and full chat session history to a managed PostgreSQL database.

---

## Challenges and Solutions

### Challenge 1: LLM API costs and quota limits
**Problem**: Both the xAI (Grok) and OpenAI APIs either had no credits or exceeded quota.
**Solution**: Switched the LLM backend to **Groq**, which provides free-tier access to Llama 3.3 70B with extremely fast inference on their custom hardware.

### Challenge 2: Slow document indexing with LightRAG
**Problem**: LightRAG's knowledge graph construction requires an LLM call per text chunk, making a 1MB PDF take 10+ minutes to index with a local model.
**Solution**: Replaced LightRAG with a lightweight custom vector similarity engine. Indexing now uses only the local embedding model (no LLM calls), reducing index time from minutes to seconds.

### Challenge 3: Gradio API breaking changes
**Problem**: The app was built targeting Gradio 4.x but Gradio 6.x was installed, which removed several parameters (`bubble_full_width`, `show_copy_button`, `show_api`) and changed the chat message format from tuples to dicts.
**Solution**: Updated all UI component calls and rewired the chat history format to use `{"role": "user/assistant", "content": "..."}` dictionaries.

### Challenge 4: Dependency conflicts
**Problem**: `pyroaring` (a LightRAG dependency) failed to build from source; `sentence-transformers 5.x` required PyTorch ≥ 2.4 but 2.2.2 was installed; NumPy 2.x broke compiled modules.
**Solution**: Installed Cython first to fix the pyroaring build, pinned `sentence-transformers<4` and `numpy<2` to stay within compatible version ranges.

### Challenge 5: HuggingFace network timeouts
**Problem**: On startup, the embedding model attempted to check HuggingFace for updates, causing 5× 10-second timeout retries and a slow start.
**Solution**: Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` in the `.env` file, forcing the library to load the already-cached model without any network check.

---

## Results

- PDF indexing: **~5 seconds** for a 1MB document
- Chat response: **streaming begins in under 1 second** (Groq API)
- Handbook generation: **~3–5 minutes** for a full 20,000-word document (14 sections × one LLM call each)
- Embedding model: **fully local**, zero API cost for retrieval

---

## Future Improvements

- Add support for multiple file formats (DOCX, TXT, web URLs)
- Implement semantic chunking instead of fixed-size chunking
- Add a progress bar during handbook generation
- Support exporting handbooks to PDF in addition to Markdown
- Add user authentication for multi-user deployments
