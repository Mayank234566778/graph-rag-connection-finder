"""CLI:  python app.py build | ask "question" | connect "A" "B" | viz"""
import sys
from graphrag import graph as g
from graphrag.retrieve import GraphRetriever
from graphrag.answer import answer

def main(argv):
    if len(argv) < 2: return print(__doc__)
    cmd = argv[1]
    if cmd == "build":
        G, chunks = g.build_graph("data/docs"); g.save(G, chunks)
        print(f"Graph built: {G.number_of_nodes()} entities, {G.number_of_edges()} relations")
    elif cmd == "ask":
        G, chunks = g.load(); r = GraphRetriever(G, chunks)
        ctx, facts, seeds = r.retrieve(" ".join(argv[2:]))
        print("Entities:", seeds); print("Facts:", *[f"\n  {a} -[{rel}]-> {b}" for a, rel, b in facts])
        print("\n" + answer(" ".join(argv[2:]), ctx, facts))
    elif cmd == "connect":
        G, _ = g.load(); path = g.connection(G, argv[2], argv[3])
        if path is None: print("Entity not found")
        elif not path: print("No connection")
        else:
            print(f"{len(path)} hop(s):")
            for u, rel, v in path: print(f"  {u} -[{rel}]-> {v}")
    elif cmd == "viz":
        G, _ = g.load(); g.export_html(G); print("Wrote graph.html")
    else: print(__doc__)

if __name__ == "__main__": main(sys.argv)
