"""
app.py — Main Gradio application for the AI Handbook Generator.

Run:
    python app.py

Environment variables (set in .env or HF Spaces secrets):
    GROQ_API_KEY     — Groq API key from console.groq.com
    SUPABASE_URL     — Your Supabase project URL (optional)
    SUPABASE_KEY     — Your Supabase anon/service key (optional)
"""
import os
import re
import uuid
import logging
from pathlib import Path

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import gradio as gr

from pdf_processor import extract_text_from_pdf, get_doc_metadata
from rag_engine import RAGEngine
from handbook_generator import HandbookGenerator
from database import Database

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Initialise shared components                                         #
# ------------------------------------------------------------------ #
rag_engine = RAGEngine()
handbook_gen = HandbookGenerator()
db = Database()

EXPORT_DIR = Path("./exports")
EXPORT_DIR.mkdir(exist_ok=True)



# ------------------------------------------------------------------ #
# Handlers                                                             #
# ------------------------------------------------------------------ #

def upload_pdfs(files, status_box):
    """Index uploaded PDFs into LightRAG and optionally Supabase."""
    if not files:
        return "⚠️ No files selected.", status_box

    messages = []
    for file_obj in files:
        path = file_obj.name
        name = Path(path).name
        try:
            # Extract text
            text = extract_text_from_pdf(path)
            meta = get_doc_metadata(path)
            word_count = len(text.split())

            if word_count < 50:
                messages.append(f"⚠️ {name} — very little text extracted ({word_count} words). Is it a scanned PDF?")
                continue

            # Index in LightRAG
            success = rag_engine.insert_document(text, name)
            if success:
                messages.append(f"✅ {name} — {meta['pages']} pages, {word_count:,} words indexed.")
            else:
                messages.append(f"❌ {name} — indexing failed. Check logs.")

            # Store metadata in Supabase (chunks)
            if db.available:
                from pdf_processor import chunk_text
                chunks = chunk_text(text)
                for i, chunk in enumerate(chunks):
                    db.store_chunk(
                        filename=name,
                        title=meta.get("title", name),
                        pages=meta["pages"],
                        chunk_index=i,
                        content=chunk,
                    )

        except Exception as e:
            messages.append(f"❌ {name} — error: {e}")
            logger.exception(f"Upload error for {name}")

    status = "\n".join(messages)
    doc_list = f"\n\n📚 Indexed documents: {', '.join(rag_engine.list_documents())}"
    return status + doc_list, status + doc_list


def chat_respond(message: str, history: list, session_id: str):
    """
    Handle a chat message. Detects handbook requests and streams output.
    Returns: updated history, cleared message box.
    """
    if not message.strip():
        return history, ""

    if not rag_engine.has_documents:
        bot_reply = "⚠️ Please upload at least one PDF document first, then ask me questions."
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": bot_reply})
        return history, ""

    # Detect handbook request — require an action verb to avoid false matches
    is_handbook = bool(re.search(
        r'\b(create|generate|write|make|build|produce)\b.{0,50}\bhandbook\b',
        message.lower()
    ))

    if is_handbook:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "⏳ Starting handbook generation…"})
        yield history, ""

        accumulated = ""
        for chunk in handbook_gen.generate_handbook(message, rag_engine):
            accumulated += chunk
            history[-1] = {"role": "assistant", "content": accumulated}
            yield history, ""

        # Save to file
        export_path = EXPORT_DIR / f"handbook_{session_id[:8]}.md"
        with open(export_path, "w", encoding="utf-8") as f:
            f.write(accumulated)

        db.save_message(session_id, "user", message)
        db.save_message(session_id, "assistant", accumulated[:5000])  # truncate for DB

    else:
        # Regular RAG query — stream tokens for instant feedback
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "▌"})
        yield history, ""

        accumulated = ""
        for token in rag_engine.query_stream(message):
            accumulated += token
            history[-1] = {"role": "assistant", "content": accumulated + "▌"}
            yield history, ""

        history[-1] = {"role": "assistant", "content": accumulated}
        db.save_message(session_id, "user", message)
        db.save_message(session_id, "assistant", accumulated)
        yield history, ""


def export_last_handbook(session_id: str):
    """Return path to the last generated handbook file."""
    path = EXPORT_DIR / f"handbook_{session_id[:8]}.md"
    if path.exists():
        return str(path), f"✅ Ready to download: {path.name}"
    return None, "⚠️ No handbook generated yet in this session."


def clear_chat():
    return [], ""


# ------------------------------------------------------------------ #
# UI                                                                   #
# ------------------------------------------------------------------ #

CUSTOM_CSS = """
:root {
  color-scheme: light !important;
  --background-fill-primary: #f9fafb !important;
  --background-fill-secondary: #f3f4f6 !important;
  --block-background-fill: #ffffff !important;
  --input-background-fill: #ffffff !important;
  --body-background-fill: #f9fafb !important;
  --chatbot-background: #f3f4f6 !important;
  --border-color-primary: #e5e7eb !important;
  --color-background-primary: #f9fafb !important;
}
.gradio-container { max-width: 1000px !important; font-family: "SF Pro Mono", ui-monospace, monospace !important; background: #f9fafb !important; }
.block, .form, .wrap { background: #ffffff !important; }
.chatbot, .chatbot .bubble-wrap { background: #f3f4f6 !important; }
.status-box { font-family: "SF Pro Mono", ui-monospace, monospace; font-size: 0.85rem; }
footer { display: none !important; }
.prose h1, .prose h2, .prose h3, .prose p, .prose li,
.prose blockquote, .prose strong, .prose em, .prose i {
  color: #111827 !important;
}
.gradio-markdown, .gradio-markdown * { color: #111827 !important; }
button[role="tab"] { color: #374151 !important; opacity: 1 !important; background: transparent !important; }
button[role="tab"][aria-selected="true"] { color: #111827 !important; background: transparent !important; }
button[role="tab"]:hover { color: #7c3aed !important; background: transparent !important; }
.file-preview, .upload-container, .file-upload, .dnd-container { background: #f9fafb !important; border: 1px dashed #9ca3af !important; }
#msg-row .gap, #msg-row > div { align-items: flex-end !important; }
#msg-row > div:last-child button { min-height: 66px !important; }
"""

def build_ui():
    with gr.Blocks(title="📖 AI Handbook Generator", css=CUSTOM_CSS, theme=gr.themes.Default(primary_hue="purple")) as demo:

        session_id = gr.State(str(uuid.uuid4()))

        # Header
        gr.Markdown(
            """# 📖 AI Handbook Generator
> Upload PDFs → Ask questions → Generate 20,000-word handbooks through conversation.
> Powered by **Llama 3.3 70B via Groq** · **sentence-transformers** · **LongWriter technique**"""
        )

        with gr.Tabs():

            # ---- Tab 1: Upload ---- #
            with gr.TabItem("📄 Upload Documents"):
                gr.Markdown(
                    "Upload one or more PDF files. The system will extract text and build "
                    "a knowledge graph using LightRAG."
                )
                file_input = gr.File(
                    label="Select PDF files",
                    file_types=[".pdf"],
                    file_count="multiple",
                )
                upload_btn = gr.Button("📥 Index Documents", variant="primary", size="lg")
                upload_status = gr.Textbox(
                    label="Indexing Status",
                    interactive=False,
                    lines=6,
                    elem_classes=["status-box"],
                )

                upload_btn.click(
                    fn=lambda: gr.update(interactive=False, value="⏳ Indexing..."),
                    outputs=upload_btn,
                    queue=False,
                ).then(
                    fn=upload_pdfs,
                    inputs=[file_input, upload_status],
                    outputs=[upload_status, upload_status],
                ).then(
                    fn=lambda: gr.update(interactive=True, value="📥 Index Documents"),
                    outputs=upload_btn,
                    queue=False,
                )

            # ---- Tab 2: Chat ---- #
            with gr.TabItem("💬 Chat & Generate Handbook"):
                gr.Markdown(
                    "Ask questions about your documents, or say **'Create a handbook on [topic]'** "
                    "to generate a 20,000+ word structured document."
                )

                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                )

                with gr.Row(elem_id="msg-row"):
                    msg_box = gr.Textbox(
                        placeholder='Ask a question or say "Create a handbook on Retrieval-Augmented Generation"',
                        label="Your message",
                        scale=5,
                        lines=2,
                    )
                    send_btn = gr.Button("Send ▶", variant="primary", scale=1, min_width=80)

                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear chat", size="sm")
                    export_btn = gr.Button("💾 Export handbook (.md)", size="sm", variant="secondary")

                with gr.Row():
                    export_file = gr.File(label="Download", scale=1)
                    export_status = gr.Textbox(label="Export status", interactive=False, lines=1, scale=2)

                # Wiring
                send_btn.click(
                    fn=lambda: gr.update(interactive=False, value="Sending..."),
                    outputs=send_btn,
                    queue=False,
                ).then(
                    fn=chat_respond,
                    inputs=[msg_box, chatbot, session_id],
                    outputs=[chatbot, msg_box],
                ).then(
                    fn=lambda: gr.update(interactive=True, value="Send ▶"),
                    outputs=send_btn,
                    queue=False,
                )
                msg_box.submit(
                    fn=lambda: gr.update(interactive=False, value="Sending..."),
                    outputs=send_btn,
                    queue=False,
                ).then(
                    fn=chat_respond,
                    inputs=[msg_box, chatbot, session_id],
                    outputs=[chatbot, msg_box],
                ).then(
                    fn=lambda: gr.update(interactive=True, value="Send ▶"),
                    outputs=send_btn,
                    queue=False,
                )
                clear_btn.click(fn=clear_chat, outputs=[chatbot, msg_box])
                export_btn.click(
                    fn=export_last_handbook,
                    inputs=[session_id],
                    outputs=[export_file, export_status],
                )

            # ---- Tab 3: Help ---- #
            with gr.TabItem("ℹ️ How to Use"):
                gr.Markdown("""
## Quick Start

### Step 1 — Configure API key
Set your Groq API key as an environment variable (or in `.env` locally):
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free key at [console.groq.com](https://console.groq.com).

Supabase is optional — the app works without it using in-memory storage.

### Step 2 — Upload & Chat
1. Go to **📄 Upload Documents** → select PDF files → click **Index Documents**
2. Go to **💬 Chat** → ask questions or request a handbook

### Handbook requests (examples)
- *"Create a handbook on Retrieval-Augmented Generation"*
- *"Generate a handbook about machine learning fundamentals"*
- *"Write a handbook covering neural network architectures"*

### How handbook generation works (LongWriter technique)
1. **Plan** — Llama 3.3 70B creates a 12-16 section table of contents with word targets
2. **Write** — Each section is generated individually (~1,500 words each) using RAG context
3. **Assemble** — All sections are joined into one 20,000+ word document

The handbook can be downloaded as a Markdown file from the **💬 Chat** tab.
""")

        def load_indexed_docs():
            docs = rag_engine.list_documents()
            if docs:
                return f"📚 Already indexed: {', '.join(docs)}"
            return ""

        demo.load(fn=load_indexed_docs, outputs=upload_status)

    return demo


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        share=False,
    )
