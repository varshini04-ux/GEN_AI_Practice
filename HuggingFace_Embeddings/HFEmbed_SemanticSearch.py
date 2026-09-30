# 10-sentence semantic search

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

my_documents = [
    "The International Space Station orbits Earth roughly every 90 minutes.",
    "A good curry balances spice, acidity, and richness in every bite.",
    "Compound interest allows savings to grow faster over long time periods.",
    "Elephants are highly social animals that live in matriarchal herds.",
    "The Renaissance was a period of major cultural and artistic rebirth in Europe.",
    "Electric vehicles rely on lithium-ion batteries for their power storage.",
    "Coral reefs support roughly a quarter of all marine species on Earth.",
    "A well-diversified portfolio spreads risk across different asset classes.",
    "The human heart beats around 100,000 times every single day.",
    "Ancient Rome built an extensive network of roads connecting its empire."
]

my_doc_embeddings = model.encode(my_documents)

def semantic_search_custom(query, docs, doc_embeddings, top_k=3):
    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, doc_embeddings)[0]
    ranked_idx = scores.argsort()[::-1][:top_k]

    print(f"Query: {query}\n")
    for idx in ranked_idx:
        print(f"Score: {scores[idx]:.3f}  |  {docs[idx]}")
    print()

semantic_search_custom("How does the body work?", my_documents, my_doc_embeddings)
semantic_search_custom("Tell me about investing and money", my_documents, my_doc_embeddings)
semantic_search_custom("What lives in the ocean?", my_documents, my_doc_embeddings)