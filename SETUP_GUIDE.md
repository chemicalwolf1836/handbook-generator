# AI Handbook Generator — Setup Guide

## Requirements
- Python 3.10+
- macOS / Linux / Windows (WSL)
- Internet connection (for Groq API calls)

---

## Step 1 — Clone the repository

```bash
git clone https://github.com/chemicalwolf1836/handbook-generator.git
cd handbook-generator
```

---

## Step 2 — Install dependencies

```bash
# Install Cython first (required to build pyroaring)
python3 -m pip install cython
python3 -m pip install pyroaring --no-build-isolation

# Install all project dependencies
python3 -m pip install -r requirements.txt

# Downgrade to compatible versions for local embeddings
python3 -m pip install "numpy<2" "sentence-transformers<4"
```

---

## Step 3 — Configure API keys

Create a `.env` file in the project folder:

```bash
cp .env.example .env
```

Then open `.env` and fill in your keys:

```
GROQ_API_KEY=your_groq_api_key_here
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_anon_key_here
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

### Getting your keys

| Key | Where to get it |
|-----|----------------|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) → API Keys → Create key (free) |
| `SUPABASE_URL` | Supabase dashboard → Settings → API → Project URL |
| `SUPABASE_KEY` | Supabase dashboard → Settings → API → anon/public key |

---

## Step 4 — Set up Supabase tables

In your Supabase dashboard, go to **SQL Editor** and run:

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

create table if not exists chat_sessions (
    id         uuid primary key default gen_random_uuid(),
    session_id text not null,
    role       text not null,
    content    text not null,
    created_at timestamptz default now()
);
```

---

## Step 5 — Run the app

```bash
python3 app.py
```

Open your browser at **http://127.0.0.1:7860**

---

## Using the app

### Upload documents
1. Go to the **📄 Upload Documents** tab
2. Select one or more PDF files
3. Click **Index Documents** — completes in seconds

### Chat
1. Go to the **💬 Chat & Generate Handbook** tab
2. Ask questions about your documents, e.g.:
   - *"What is the main topic of this document?"*
   - *"Summarise the key findings"*

### Generate a handbook
Type any of the following in the chat:
- *"Create a handbook on Retrieval-Augmented Generation"*
- *"Generate a handbook about machine learning"*
- *"Write a handbook covering the topics in these documents"*

The generator will plan, write, and assemble a 20,000+ word handbook, then make it available to download as a `.md` file.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `indexing failed` | Check your `GROQ_API_KEY` is set correctly in `.env` |
| App hangs on startup | Ensure `HF_HUB_OFFLINE=1` is in `.env` |
| Supabase errors | Run the SQL setup in Step 4; Supabase is optional — the app works without it |
| Port already in use | Run `lsof -ti :7860 \| xargs kill -9` then restart |
