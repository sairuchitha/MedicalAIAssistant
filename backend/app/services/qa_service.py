from app.config import settings
from app.services.llm_client import generate_with_llm

LOOKUP_KEYWORDS = ["last", "current", "most recent", "what is", "when was", "what are"]
REASONING_KEYWORDS = ["trend", "over time", "history", "compare", "progression", "worsening", "changing"]


def classify_question(question: str) -> str:
    q = question.lower()
    for kw in LOOKUP_KEYWORDS:
        if kw in q:
            return "lookup"
    for kw in REASONING_KEYWORDS:
        if kw in q:
            return "reasoning"
    return "reasoning"


def build_prompt(question: str, retrieved_chunks, qtype: str) -> str:
    selected = retrieved_chunks[:settings.LOOKUP_TOP_K] if qtype == "lookup" else retrieved_chunks[:settings.REASONING_TOP_K]
    if qtype == "reasoning":
        selected = sorted(selected, key=lambda x: x.get("date") or "")
    notes_block = "\n\n".join(
        f"[{i+1}] Date: {c['date']} | Type: {c['note_type']} | Section: {c['section_name']} | NoteID: {c['note_id']}\n{c['text']}"
        for i, c in enumerate(selected)
    )
    if qtype == "lookup":
        return f"""
Answer the following clinical question using ONLY the retrieved notes below.
Be specific. Include values, dates, and units if available.
If the answer is not explicitly present, say: Not documented in available records.

Question:
{question}

Retrieved Notes:
{notes_block}
""".strip()
    return f"""
Using ONLY the retrieved clinical notes below, synthesize an answer across visits.
Do not infer beyond documented evidence.
Acknowledge if evidence is incomplete.

Question:
{question}

Retrieved Notes:
{notes_block}
""".strip()


def answer_question(question: str, retrieved_chunks, question_type: str = None):
    # Accept a pre-classified type from the API layer (avoids running classifier twice)
    qtype = question_type if question_type is not None else classify_question(question)
    prompt = build_prompt(question, retrieved_chunks, qtype)
    answer = generate_with_llm(prompt)
    selected = retrieved_chunks[:3] if qtype == "lookup" else retrieved_chunks[:5]
    citations = [
        {
            "id": i + 1,
            "date": str(c["date"]),
            "note_type": str(c["note_type"]),
            "section_name": str(c["section_name"]),
            "note_id": str(c["note_id"]),
        }
        for i, c in enumerate(selected)
    ]
    return {"question_type": qtype, "answer": answer, "citations": citations}
