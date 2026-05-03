"""
Handbook Generator — implements the LongWriter / AgentWrite technique.

The core idea (from the research paper):
  1. PLAN  — Ask the LLM to create a detailed table of contents with a
             target word count per section (summing to 20,000+).
  2. WRITE — Generate each section sequentially. Pass the section heading,
             its target word count, relevant RAG context, and a running
             summary of what has been written so far.
  3. ASSEMBLE — Join all sections into a single document.

This avoids the typical LLM context-window output limit by breaking the
generation into multiple API calls, each producing ~1,000–2,000 words.
"""
import os
import json
import logging
import re
from typing import Generator

from openai import OpenAI

logger = logging.getLogger(__name__)

# Target total word count
TARGET_WORDS = 20_000
# Words per section (approximately)
WORDS_PER_SECTION = 1_500


def _llm_client() -> OpenAI:
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        return OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
    return OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")

_OLLAMA_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile") if os.getenv("GROQ_API_KEY") else os.getenv("OLLAMA_MODEL", "llama3.2")


def _count_words(text: str) -> int:
    return len(text.split())


# ------------------------------------------------------------------ #
# Phase 1 – Plan                                                       #
# ------------------------------------------------------------------ #

PLAN_SYSTEM = """You are an expert technical writer and curriculum designer.
Your task is to create a detailed, professional table of contents for a
comprehensive handbook. The handbook must be AT LEAST 20,000 words in total.

Return ONLY a valid JSON array (no markdown fences, no commentary) with this structure:
[
  {
    "section_number": "1",
    "title": "Introduction",
    "subsections": ["1.1 Background", "1.2 Purpose"],
    "target_words": 1200,
    "key_topics": ["overview", "goals", "audience"]
  },
  ...
]

Guidelines:
- Include 12–16 top-level sections.
- Each section should target 1,200–2,000 words.
- Include subsections for depth and structure.
- Cover the topic exhaustively: theory, practice, tools, case studies, best practices, future directions.
- Total target_words across all sections must be ≥ 20,000.
"""

def plan_handbook(topic: str, rag_context: str, client: OpenAI) -> list[dict]:
    """Generate a structured table of contents for the handbook."""
    user_prompt = f"""Topic: {topic}

Context from uploaded documents:
{rag_context[:3000]}

Create a comprehensive 20,000+ word handbook table of contents on this topic.
The handbook should synthesise the provided context and expand with expert knowledge."""

    response = client.chat.completions.create(
        model=_OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": PLAN_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=2000,
        temperature=0.3,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if present
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"```$", "", raw).strip()

    try:
        plan = json.loads(raw)
        logger.info(f"Plan created: {len(plan)} sections, "
                    f"~{sum(s.get('target_words', 1500) for s in plan):,} target words")
        return plan
    except json.JSONDecodeError as e:
        logger.error(f"Plan JSON parse error: {e}\nRaw: {raw[:500]}")
        # Return a sensible fallback plan
        return _fallback_plan(topic)


def _fallback_plan(topic: str) -> list[dict]:
    """Minimal fallback if the LLM returns malformed JSON."""
    sections = [
        "Introduction and Overview",
        "Foundational Concepts",
        "Core Principles and Theory",
        "Architecture and Design",
        "Implementation Guide",
        "Tools and Technologies",
        "Best Practices",
        "Advanced Techniques",
        "Case Studies and Examples",
        "Challenges and Solutions",
        "Performance and Optimisation",
        "Security Considerations",
        "Future Trends",
        "Summary and Conclusion",
        "Glossary",
        "References and Further Reading",
    ]
    return [
        {
            "section_number": str(i + 1),
            "title": title,
            "subsections": [],
            "target_words": 1300,
            "key_topics": [topic],
        }
        for i, title in enumerate(sections)
    ]


# ------------------------------------------------------------------ #
# Phase 2 – Write (section by section)                                 #
# ------------------------------------------------------------------ #

WRITE_SYSTEM = """You are an expert technical writer producing a professional handbook.
Write detailed, authoritative, well-structured content.

Rules:
- Write ONLY the content for the specified section — no titles beyond what's asked.
- Target the word count given. Write thoroughly to meet it.
- Use proper markdown: ## for main heading, ### for subsections, **bold** for terms.
- Integrate information from the provided context where relevant.
- Add concrete examples, explanations, and expert insights.
- Do NOT include phrases like "In this section we will..." — just write the content directly.
"""


def write_section(
    section: dict,
    topic: str,
    rag_context: str,
    previous_summary: str,
    client: OpenAI,
) -> str:
    """Generate one section of the handbook."""
    subsection_str = ""
    if section.get("subsections"):
        subsection_str = "\nSubsections to cover:\n" + "\n".join(
            f"  - {s}" for s in section["subsections"]
        )

    user_prompt = f"""Handbook topic: {topic}
Section {section['section_number']}: {section['title']}
Target word count: {section.get('target_words', 1500)} words
Key topics: {', '.join(section.get('key_topics', []))}
{subsection_str}

--- CONTEXT FROM UPLOADED DOCUMENTS ---
{rag_context[:2500]}

--- WHAT HAS BEEN WRITTEN SO FAR (summary) ---
{previous_summary[:1500] if previous_summary else 'This is the first section.'}

Write the full content for this section now. Begin with the section heading (## {section['section_number']}. {section['title']}).
Write approximately {section.get('target_words', 1500)} words."""

    response = client.chat.completions.create(
        model=_OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": WRITE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=3000,
        temperature=0.5,
    )

    return response.choices[0].message.content.strip()


def summarise_progress(sections_so_far: list[str], client: OpenAI) -> str:
    """Generate a brief running summary to pass to the next section."""
    combined = "\n\n".join(sections_so_far[-3:])  # last 3 sections only
    response = client.chat.completions.create(
        model=_OLLAMA_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Summarise the key points covered so far in 150 words.",
            },
            {"role": "user", "content": combined[:4000]},
        ],
        max_tokens=200,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


# ------------------------------------------------------------------ #
# Main generator                                                        #
# ------------------------------------------------------------------ #

class HandbookGenerator:
    def __init__(self):
        self.client: OpenAI | None = None
        try:
            self.client = _llm_client()
        except Exception as e:
            logger.error(f"Could not initialise LLM client: {e}")

    def generate_handbook(
        self, request: str, rag_engine
    ) -> Generator[str, None, None]:
        """
        Generator that yields progress updates and finally the full handbook.
        Usage: for chunk in gen.generate_handbook(request, rag_engine): print(chunk)
        """
        if not self.client:
            yield "❌ XAI_API_KEY not configured."
            return

        if not rag_engine.has_documents:
            yield "⚠️ No documents indexed. Please upload at least one PDF first."
            return

        # Extract topic from the request
        topic = self._extract_topic(request)
        yield f"📋 **Topic detected:** {topic}\n\n"

        # --- Phase 1: Plan ---
        yield "🗂️ **Phase 1/3 — Planning handbook structure...**\n\n"
        rag_overview = rag_engine.get_context(topic)
        plan = plan_handbook(topic, rag_overview, self.client)
        total_target = sum(s.get("target_words", 1500) for s in plan)

        toc_lines = [f"# {topic} — Handbook\n", "## Table of Contents\n"]
        for sec in plan:
            toc_lines.append(f"{sec['section_number']}. {sec['title']} (~{sec.get('target_words',1500):,} words)")
            for sub in sec.get("subsections", []):
                toc_lines.append(f"   - {sub}")
        toc_lines.append(f"\n**Total target: ~{total_target:,} words across {len(plan)} sections**\n")

        yield "\n".join(toc_lines) + "\n\n---\n\n"

        # --- Phase 2: Write ---
        yield f"✍️ **Phase 2/3 — Writing {len(plan)} sections...**\n\n"

        all_sections: list[str] = []
        total_words = 0
        running_summary = ""

        for i, section in enumerate(plan):
            sec_title = f"{section['section_number']}. {section['title']}"
            yield f"  ⏳ Writing section {i+1}/{len(plan)}: **{sec_title}**...\n"

            # Get focused context for this section
            section_context = rag_engine.get_context(
                f"{topic} {section['title']} {' '.join(section.get('key_topics', []))}"
            )

            content = write_section(
                section=section,
                topic=topic,
                rag_context=section_context,
                previous_summary=running_summary,
                client=self.client,
            )
            all_sections.append(content)
            words = _count_words(content)
            total_words += words
            yield f"  ✅ Section {i+1} done — {words:,} words (running total: {total_words:,})\n"

            # Update running summary every 3 sections
            if (i + 1) % 3 == 0:
                running_summary = summarise_progress(all_sections, self.client)

        # --- Phase 3: Assemble ---
        yield f"\n📖 **Phase 3/3 — Assembling handbook ({total_words:,} words)...**\n\n"

        header = f"# {topic}\n## A Comprehensive Handbook\n\n"
        header += f"*Generated from {rag_engine.document_count} uploaded document(s). "
        header += f"Total: ~{total_words:,} words.*\n\n---\n\n"

        toc_block = "\n".join(toc_lines) + "\n\n---\n\n"
        handbook = header + toc_block + "\n\n".join(all_sections)

        yield f"\n\n{'='*60}\n\n"
        yield handbook
        yield f"\n\n{'='*60}\n"
        yield f"✅ **Handbook complete! {total_words:,} words across {len(plan)} sections.**\n"

        if total_words < TARGET_WORDS:
            yield f"⚠️ Note: {TARGET_WORDS - total_words:,} words short of the 20,000 target. "
            yield "Consider uploading more source documents or requesting additional sections.\n"

    def _extract_topic(self, request: str) -> str:
        """Pull the handbook topic out of the user's chat message."""
        # Common patterns: "Create a handbook on X", "Generate a handbook about X"
        patterns = [
            r"handbook (?:on|about|for|covering)\s+(.+)",
            r"create (?:a )?handbook\s+(?:on|about|for)?\s*(.+)",
            r"generate (?:a )?handbook\s+(?:on|about|for)?\s*(.+)",
            r"write (?:a )?handbook\s+(?:on|about|for)?\s*(.+)",
        ]
        for pat in patterns:
            m = re.search(pat, request, re.IGNORECASE)
            if m:
                return m.group(1).strip().rstrip("?.")
        # Fallback: use the whole request
        return request.replace("handbook", "").strip() or "the uploaded documents"
