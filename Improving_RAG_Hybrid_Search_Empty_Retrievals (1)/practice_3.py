import re
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

documents = {
    "ai_overview.txt": """Artificial Intelligence: An Overview

Artificial Intelligence, or AI, is the field of computer science focused on building systems
that can perform tasks which normally require human intelligence.

Machine Learning is a subset of AI where systems learn patterns from data instead of following
hardcoded rules.

Deep Learning takes Machine Learning further by using neural networks with many layers to
automatically discover complex patterns in data.

Retrieval-Augmented Generation, or RAG, combines a search system with a language model so answers
are grounded in real retrieved documents instead of only the model's memorized training data.
""",
    "embeddings_in_practice.txt": """Embeddings and Vector Search in Practice

Modern search engines increasingly rely on embeddings rather than simple keyword matching, allowing
systems to find results that match the meaning of a query, not just its exact words.

Vector databases such as ChromaDB store these embeddings and allow fast similarity search across
thousands or millions of documents at once.
""",
    "product_catalog.txt": """Product Catalog Notes

The wireless noise-cancelling headphones are listed under product code SKU-4521 and are currently
our best-selling audio accessory.

The standing desk converter is listed under product code SKU-7788 and ships within three business
days.

Return requests for any product must reference the exact SKU code so our warehouse team can locate
the correct item quickly.
""",
}

splitter = RecursiveCharacterTextSplitter(
    chunk_size=220, chunk_overlap=30, separators=["\n\n", "\n", ". ", " ", ""]
)
client = chromadb.Client()
embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.create_collection(name="rag_hybrid_demo", embedding_function=embedder)

all_chunks, all_chunk_meta = [], []
for source, text in documents.items():
    doc_chunks = splitter.split_text(text)
    meta = [{"source": source, "chunk_index": i} for i in range(len(doc_chunks))]
    collection.add(
        documents=doc_chunks,
        metadatas=meta,
        ids=[f"{source}_chunk_{i}" for i in range(len(doc_chunks))],
    )
    all_chunks.extend(doc_chunks)
    all_chunk_meta.extend(meta)
print("Total chunks stored:", collection.count())

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "am", "of", "in", "on", "at",
    "to", "for", "from", "by", "with", "and", "or", "but", "if", "it", "its", "this", "that",
    "these", "those", "what", "whats", "which", "who", "whom", "how", "why", "when", "where",
    "do", "does", "did", "can", "could", "should", "would", "will", "me", "my", "tell", "about",
    "s", "i", "you", "we", "they", "there", "any", "some",
}


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def remove_stopwords(query):
    return " ".join(t for t in tokenize(query) if t not in STOPWORDS)


bm25 = BM25Okapi([tokenize(c) for c in all_chunks])


def semantic_search(query, n_results=10, min_similarity=0.2):
    res = collection.query(query_texts=[query], n_results=n_results,
                           include=["documents", "metadatas", "distances"])
    out = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        sim = 1 - dist
        if sim >= min_similarity:
            out.append({"text": doc, "source": meta["source"], "similarity": round(sim, 4)})
    return out


def bm25_search(query, n_results=10):
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]
    return [
        {"text": all_chunks[i], "source": all_chunk_meta[i]["source"], "bm25_score": round(float(scores[i]), 4)}
        for i in ranked if scores[i] > 0
    ]


def hybrid_search(query, n_results=3, k=60):
    rrf, lookup = {}, {}
    for rank, r in enumerate(semantic_search(query)):
        rrf[r["text"]] = rrf.get(r["text"], 0) + 1 / (k + rank + 1)
        lookup[r["text"]] = r
    for rank, r in enumerate(bm25_search(query)):
        rrf[r["text"]] = rrf.get(r["text"], 0) + 1 / (k + rank + 1)
        lookup.setdefault(r["text"], r)
    ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:n_results]
    return [{"text": t, "source": lookup[t]["source"], "rrf_score": round(s, 5)} for t, s in ranked]


def retrieve_with_threshold(query, n_results=5, similarity_threshold=0.3):
    res = collection.query(query_texts=[query], n_results=n_results,
                           include=["documents", "metadatas", "distances"])
    out = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        sim = 1 - dist
        if sim >= similarity_threshold:
            out.append({"text": doc, "source": meta["source"], "similarity": round(sim, 4)})
    return out


KNOWLEDGE_BASE_TOPICS = sorted({m["source"] for m in all_chunk_meta})



def smart_retrieve(query, n_results=4, similarity_threshold=0.3):
    # Step 1: semantic search with a quality threshold
    primary = retrieve_with_threshold(query, n_results, similarity_threshold)
    if primary:
        return {"status": "ok", "method": "semantic", "chunks": primary}

    
    hybrid = hybrid_search(query, n_results=n_results)
    if hybrid:
        return {"status": "ok", "method": "hybrid_fallback", "chunks": hybrid}

    
    cleaned = remove_stopwords(query)
    if cleaned and cleaned != query.lower():
        retry = hybrid_search(cleaned, n_results=n_results)
        if retry:
            return {"status": "ok", "method": f"stopword_retry ('{cleaned}')", "chunks": retry}

    
    return {
        "status": "empty", "method": None, "chunks": [],
        "message": "I don't know based on the provided documents. "
                   f"This knowledge base currently covers: {', '.join(KNOWLEDGE_BASE_TOPICS)}.",
    }


tests = [
    "What is deep learning?",
    "Tell me about SKU-4521",
    "what is the price of the SKU-7788",
    "chocolate cake recipe",
]
for q in tests:
    result = smart_retrieve(q)
    print(f"\nQuery: {q}")
    print("  cleaned query:", remove_stopwords(q))
    print("  status:", result["status"], "|", result["method"])
    if result["status"] == "empty":
        print("  message:", result["message"])
    for c in result["chunks"][:2]:
        print("   ->", c["text"][:70].replace("\n", " "))
