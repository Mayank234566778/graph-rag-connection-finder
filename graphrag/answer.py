import os

def answer(question, chunks, facts):
    ctx = "\n".join(c["text"] for c in chunks)
    fx = "\n".join(f"({a}) -[{r}]-> ({b})" for a, r, b in facts)
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            msg = anthropic.Anthropic().messages.create(
                model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"), max_tokens=600,
                messages=[{"role": "user", "content":
                    f"Answer using ONLY the context and graph facts.\n\nContext:\n{ctx}\n\n"
                    f"Graph facts:\n{fx}\n\nQuestion: {question}"}])
            return msg.content[0].text
        except Exception as e:
            print(f"[warn] LLM answer failed ({e})")
    return "(extractive mode - set ANTHROPIC_API_KEY for generated answers)\n" + ctx
