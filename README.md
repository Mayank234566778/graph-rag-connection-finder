# Graph RAG – Connection Finder

Retrieval-Augmented Generation over a **knowledge graph**: extracts entities and relations from documents, builds a graph, and answers questions / finds multi-hop connections that plain vector RAG misses.

## How it works
1. **Extract** – (subject, relation, object) triples via Claude (if `ANTHROPIC_API_KEY` set) or a rule-based fallback.
2. **Graph** – NetworkX MultiDiGraph; each edge links back to its source chunk.
3. **Hybrid retrieval** – TF-IDF similarity + graph-neighbourhood boost around entities in the question.
4. **Connections** – shortest path between any two entities, with relation labels.
5. **Answer** – Claude grounded on chunks + graph facts (extractive fallback offline).

## Run
```bash
pip install -r requirements.txt
python app.py build
python app.py connect "Karan Singh" "HDFC Bank"
python app.py ask "Who founded NeuraLabs?"
python app.py viz        # writes interactive graph.html
pytest
```
Add your own `.txt` files to `data/docs/`, then rebuild.

## Example
`Karan Singh -[works at]-> Sequoia Capital -[invested in]-> NeuraLabs <-[signed a contract with]- HDFC Bank`

