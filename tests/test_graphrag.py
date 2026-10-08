from graphrag import graph as g
from graphrag.retrieve import GraphRetriever

def setup():
    return g.build_graph("data/docs", use_llm=False)

def test_graph_built():
    G, _ = setup(); assert "NeuraLabs" in G and G.number_of_edges() > 5

def test_connection_multi_hop():
    G, _ = setup(); p = g.connection(G, "Rohan Verma", "Priya Sharma")
    assert p and len(p) >= 2

def test_retrieval_uses_graph():
    G, ch = setup(); ctx, facts, seeds = GraphRetriever(G, ch).retrieve("Who founded NeuraLabs?")
    assert "NeuraLabs" in seeds and facts
