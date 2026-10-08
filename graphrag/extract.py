"""Triple extraction: (subject, relation, object) from text.
Uses Claude if ANTHROPIC_API_KEY is set, else a rule-based fallback."""
import json, os, re

ENTITY = re.compile(r"\b(?:[A-Z][\w&.-]*)(?:\s+[A-Z][\w&.-]*)*")
STOP = {"The", "A", "An", "In", "On", "At", "He", "She", "They", "It", "After",
        "Before", "Later", "Then", "This", "That", "Her", "His", "Their", "As", "By", "Also"}

def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

def _clean(e):
    words = e.split()
    while words and words[0] in STOP:
        words.pop(0)
    return " ".join(words).strip(" .,")

def rule_based(sentence):
    ents = [(m.start(), m.end(), _clean(m.group())) for m in ENTITY.finditer(sentence)]
    ents = [e for e in ents if len(e[2]) > 1]
    triples = []
    for (s1, e1, a), (s2, e2, b) in zip(ents, ents[1:]):
        rel = sentence[e1:s2].strip(" ,;:").lower()
        if a != b and 0 < len(rel.split()) <= 6:
            triples.append((a, rel, b))
    return triples

def llm_extract(text):
    import anthropic
    msg = anthropic.Anthropic().messages.create(
        model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"), max_tokens=1000,
        messages=[{"role": "user", "content":
            "Extract knowledge-graph triples. Return ONLY a JSON list of "
            "[subject, relation, object] with short lowercase relations.\n\n" + text}])
    txt = re.sub(r"```json|```", "", msg.content[0].text).strip()
    return [tuple(t) for t in json.loads(txt) if len(t) == 3]

def extract_triples(text, use_llm=None):
    use_llm = bool(os.getenv("ANTHROPIC_API_KEY")) if use_llm is None else use_llm
    if use_llm:
        try:
            return llm_extract(text)
        except Exception as e:
            print(f"[warn] LLM extraction failed ({e}); using rules")
    out = []
    for s in split_sentences(text):
        out.extend(rule_based(s))
    return out
