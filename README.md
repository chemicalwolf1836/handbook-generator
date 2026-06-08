---
title: AI Handbook Generator
emoji: 📖
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "5.0.0"
app_file: app.py
pinned: false
python_version: "3.11"
---

> ▶️ **Live version (reimplemented, in-browser):**
> https://batmagnai-ganbaatar-portfolio.vercel.app/handbook-generator-app.html

# 📖 AI Handbook Generator

> Upload PDFs → Ask questions → Generate 20,000-word handbooks through conversation.

Built for the LunarTech AI Engineering Assignment.

---

## What It Does

| Feature | Description |
|---|---|
| PDF Upload | Upload any text-based PDF; system extracts and indexes content |
| Smart Chat | Ask questions; get answers grounded in your uploaded documents |
| Handbook Generation | Request a handbook on any topic; receive a 20,000+ word structured document |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    YOUR APPLICATION                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  📄 PDF Upload → 🧠 LightRAG → 💬 Chat UI → 📖 Handbook    │
│    (pdfplumber)   (Supabase)   (Gradio)   (Grok 4.1)       │
│                               (20k words via LongWriter)    │
└─────────────────────────────────────────────────────────────┘
```

### Components

| Component | Technology | Purpose |
|---|---|---|
| Frontend | Gradio | Chat interface with file upload |
| LLM | Grok 4.1 (via xAI API) | Long-context generation |
| RAG | LightRAG | Knowledge graph from PDFs |
| Database | Supabase (pgvector) | Vector storage & chat history |
| PDF Parser | pdfplumber + pypdf | Text extraction |

---

## How the LongWriter Technique Works

Generating 20,000+ words in a single LLM call exceeds typical output limits.
The **AgentWrite** approach (from the LongWriter research paper) solves this:

1. **Plan** — Grok 4.1 creates a detailed table of contents (12-16 sections, each with a word-count target totalling 20,000+)
2. **Write** — Each section is generated in a separate API call, using:
   - LightRAG context relevant to that section's topics
   - A rolling summary of previously written sections
   - The section's target word count
3. **Assemble** — All sections are concatenated into the final document

This produces coherent, well-structured long-form content without hitting token limits.

---

## Setup

### 1. Clone / copy this project

```bash
git clone <your-repo-url>
cd handbook-generator
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API keys

```bash
cp .env.example .env
# Edit .env and fill in your keys
```

Required keys:
- **XAI_API_KEY** — Get from [console.x.ai](https://console.x.ai/)
- **SUPABASE_URL** + **SUPABASE_KEY** — From your [Supabase](https://supabase.com) project dashboard → Settings → API

### 4. Set up Supabase database

In your Supabase project, go to **SQL Editor** and run:

```sql
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
    role       text not null,
    content    text not null,
    created_at timestamptz default now()
);
```

### 5. Run the app

```bash
python app.py
```

Open [http://localhost:7860](http://localhost:7860) in your browser.

---

## Usage

### Upload documents
1. Go to **📄 Upload Documents** tab
2. Click to select one or more PDF files
3. Click **Index Documents** — wait for confirmation

### Ask questions
In the **💬 Chat** tab, ask anything about your documents:
> *"What are the main findings of the paper?"*
> *"Explain the architecture described in section 3."*

### Generate a handbook
In the **💬 Chat** tab, request a handbook:
> *"Create a handbook on Retrieval-Augmented Generation"*
> *"Generate a handbook about the topics covered in these papers"*

Generation takes 5–15 minutes depending on document size. Progress is streamed live.
When done, click **💾 Export handbook (.md)** to download.

---

## Test Case

**Input:** Upload 2-3 AI-related research papers  
**Chat:** `"Create a handbook on Retrieval-Augmented Generation"`  
**Expected output:**
- Table of contents (12-16 sections)
- 20,000+ word structured document
- Content grounded in uploaded papers
- Downloadable as Markdown

---

## File Structure

```
handbook-generator/
├── app.py                  # Main Gradio app (entry point)
├── pdf_processor.py        # PDF text extraction & chunking
├── rag_engine.py           # LightRAG knowledge graph wrapper
├── handbook_generator.py   # LongWriter handbook generation
├── database.py             # Supabase integration
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── .env                    # Your actual keys (git-ignored)
├── lightrag_storage/       # LightRAG graph data (auto-created)
└── exports/                # Generated handbooks (auto-created)
```

---

## Submission Checklist

- [x] Working application (local)
- [x] PDF upload and text extraction
- [x] LightRAG knowledge graph indexing
- [x] Chat interface with contextual responses
- [x] 20,000-word handbook generation via chat (LongWriter technique)
- [x] Supabase vector storage integration
- [x] Markdown export of generated handbooks
- [ ] GitHub repository (push to GitHub before submitting)
- [ ] Demo video / screenshots
- [ ] Submit to: tk.lunartech@gmail.com
