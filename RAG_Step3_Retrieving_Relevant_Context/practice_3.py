import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

documents = {
    "ai_overview.txt": """Artificial Intelligence: An Overview

Artificial Intelligence, or AI, is the field of computer science focused on building systems
that can perform tasks which normally require human intelligence. These tasks include
understanding language, recognizing images, making decisions, and learning from experience.

Machine Learning is a subset of AI where systems learn patterns from data instead of following
hardcoded rules. Instead of programming every decision by hand, we feed the system many examples,
and it learns the underlying pattern on its own.

Deep Learning takes Machine Learning further by using neural networks with many layers. These
layered networks can automatically discover complex patterns in large amounts of data, such as
recognizing faces in photos or understanding the meaning of a sentence.

Retrieval-Augmented Generation, or RAG, is a technique that combines a search system with a
language model. Instead of relying only on what a language model memorized during training, RAG
first retrieves relevant, up-to-date information from a document collection, then passes that
information to the model so it can generate a more accurate, grounded answer.""",
    "embeddings_in_practice.txt": """Embeddings and Vector Search in Practice

Modern search engines increasingly rely on embeddings rather than simple keyword matching.
An embedding model converts a search query into the same vector space as the stored documents,
allowing the system to find results that match the meaning of the query, not just its exact words.

This is especially powerful for customer support systems, where a user might phrase a question very
differently from how it appears in a help article, yet still expect to find the right answer.

Vector databases such as ChromaDB store these embeddings and allow fast similarity search across
thousands or millions of documents at once."""
}

splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=40, separators=["\n\n", "\n", ". ", " ", ""])
client = chromadb.Client()
embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.create_collection(name="rag_chunks", embedding_function=embedder)

for source, text in documents.items():
    doc_chunks = splitter.split_text(text)
    collection.add(
        documents=doc_chunks,
        metadatas=[{"source": source, "chunk_index": i} for i in range(len(doc_chunks))],
        ids=[f"{source}_chunk_{i}" for i in range(len(doc_chunks))],
    )
print("Chunks before:", collection.count())


def distance_to_similarity(distance):
    return 1 - distance


def retrieve_context(query, collection, n_results=5, similarity_threshold=0.25, source_filter=None):
    query_kwargs = {
        "query_texts": [query],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if source_filter:
        query_kwargs["where"] = {"source": source_filter}

    results = collection.query(**query_kwargs)

    context_chunks = []
    for doc, meta, distance in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        similarity = distance_to_similarity(distance)
        if similarity >= similarity_threshold:
            context_chunks.append({
                "text": doc,
                "source": meta["source"],
                "chunk_index": meta["chunk_index"],
                "similarity": round(similarity, 4),
            })
    return context_chunks



third_source = "transformers_and_llms.txt"
third_text = """Transformers and Large Language Models

The transformer is a neural network architecture that uses a mechanism called self-attention
to weigh the importance of different words in a sentence. Unlike older recurrent networks,
transformers process all words in parallel, which makes training much faster.

Large Language Models, or LLMs, are transformers trained on huge amounts of text. They can
write essays, answer questions, translate languages, and generate code."""

third_chunks = splitter.split_text(third_text)
collection.add(
    documents=third_chunks,
    metadatas=[{"source": third_source, "chunk_index": i} for i in range(len(third_chunks))],
    ids=[f"{third_source}_chunk_{i}" for i in range(len(third_chunks))],
)
print("Chunks after:", collection.count())


for q in ["What is self-attention in a transformer?", "What can large language models do?"]:
    print(f"\nQuery: {q}")
    for c in retrieve_context(q, collection, n_results=3):
        print(f"  [{c['similarity']}] ({c['source']}) {c['text'][:70]}...")


print("\nFiltered to the third document only:")
for c in retrieve_context("how do models process words", collection, source_filter=third_source):
    print(f"  [{c['similarity']}] {c['text'][:70]}...")