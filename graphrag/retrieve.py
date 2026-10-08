"""Hybrid retrieval: TF-IDF vector search + graph neighbourhood expansion."""
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class GraphRetriever:
    def __init__(self, G, chunks, hops=1):
        self.G, self.chunks, self.hops = G, chunks, hops
        self.vec = TfidfVectorizer(stop_words="english").fit([c["text"] for c in chunks])
        self.mat = self.vec.transform([c["text"] for c in chunks])

    def seed_entities(self, question):
        q = question.lower()
        return [n for n in self.G if n.lower() in q]

    def retrieve(self, question, k=3):
        sims = cosine_similarity(self.vec.transform([question]), self.mat)[0]
        scores = {i: float(s) for i, s in enumerate(sims) if s > 0}
        seeds = self.seed_entities(question)
        U, reached = self.G.to_undirected(as_view=True), set()
        for s in seeds:
            reached |= set(nx.single_source_shortest_path_length(U, s, cutoff=self.hops))
        for c in self.chunks:  # graph boost
            o = len(reached & set(c["entities"]))
            if o: scores[c["id"]] = scores.get(c["id"], 0) + 0.15 * o
        top = sorted(scores, key=scores.get, reverse=True)[:k]
        facts = []
        for s in seeds:
            for u, v, d in list(self.G.out_edges(s, data=True)) + list(self.G.in_edges(s, data=True)):
                if (u, d["relation"], v) not in facts: facts.append((u, d["relation"], v))
        return [self.chunks[i] for i in top], facts[:15], seeds
