"""Build, save and query the knowledge graph."""
import json, pathlib
import networkx as nx
from networkx.readwrite import json_graph
from .extract import extract_triples, split_sentences

def load_docs(folder):
    return {p.name: p.read_text() for p in sorted(pathlib.Path(folder).glob("*.txt"))}

def chunk(text, size=3):
    s = split_sentences(text)
    return [" ".join(s[i:i + size]) for i in range(0, len(s), size)]

def build_graph(folder, use_llm=None):
    G, chunks = nx.MultiDiGraph(), []
    for doc, text in load_docs(folder).items():
        for c in chunk(text):
            cid = len(chunks)
            chunks.append({"id": cid, "doc": doc, "text": c, "entities": []})
            for a, rel, b in extract_triples(c, use_llm):
                for n in (a, b):
                    G.add_node(n)
                    if n not in chunks[cid]["entities"]:
                        chunks[cid]["entities"].append(n)
                G.add_edge(a, b, relation=rel, chunk=cid, doc=doc)
    return G, chunks

def save(G, chunks, path="graph.json"):
    pathlib.Path(path).write_text(json.dumps(
        {"graph": json_graph.node_link_data(G, edges="edges"), "chunks": chunks}, indent=1))

def load(path="graph.json"):
    d = json.loads(pathlib.Path(path).read_text())
    return json_graph.node_link_graph(d["graph"], edges="edges"), d["chunks"]

def find_entity(G, query):
    q = query.lower().strip()
    exact = [n for n in G if n.lower() == q]
    if exact: return exact[0]
    part = sorted((n for n in G if q in n.lower()), key=len)
    return part[0] if part else None

def connection(G, a, b):
    """Shortest path between two entities (direction-agnostic) with edge labels."""
    a, b = find_entity(G, a), find_entity(G, b)
    if not a or not b: return None
    try:
        path = nx.shortest_path(G.to_undirected(as_view=True), a, b)
    except nx.NetworkXNoPath:
        return []
    hops = []
    for u, v in zip(path, path[1:]):
        if G.has_edge(u, v):
            hops.append((u, G[u][v][next(iter(G[u][v]))]["relation"], v))
        else:
            hops.append((u, "<-" + G[v][u][next(iter(G[v][u]))]["relation"] + "-", v))
    return hops

def export_html(G, path="graph.html"):
    from pyvis.network import Network
    net = Network(height="750px", width="100%", directed=True, cdn_resources="in_line")
    for n in G: net.add_node(n, label=n, size=10 + 4 * G.degree(n))
    for u, v, d in G.edges(data=True): net.add_edge(u, v, label=d["relation"])
    net.write_html(path)
