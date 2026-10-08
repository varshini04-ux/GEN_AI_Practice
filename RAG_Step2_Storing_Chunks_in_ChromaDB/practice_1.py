# Exercise 1: Add a third document and confirm collection.count()
# Standalone: run with  python add_third_document.py
# Install once: pip install langchain-text-splitters chromadb sentence-transformers

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------- Setup ----------
article_text = """Artificial Intelligence: An Overview

Artificial Intelligence, or AI, is the field of computer science focused on building systems
that can perform tasks which normally require human intelligence. These tasks include
understanding language, recognizing images, making decisions, and learning from experience.

Machine Learning is a subset of AI where systems learn patterns from data instead of following
hardcoded rules. Instead of programming every decision by hand, we feed the system many examples,
and it learns the underlying pattern on its own. This approach powers everything from spam filters
to recommendation systems.

Deep Learning takes Machine Learning further by using neural networks with many layers. These
layered networks can automatically discover complex patterns in large amounts of data, such as
recognizing faces in photos or understanding the meaning of a sentence.

Natural Language Processing, or NLP, is the branch of AI that focuses specifically on human
language. NLP powers chatbots, translation tools, and search engines. A key building block of
modern NLP is the embedding, which converts words or sentences into numeric vectors that capture
meaning.

Retrieval-Augmented Generation, or RAG, is a technique that combines a search system with a
language model. Instead of relying only on what a language model memorized during training, RAG
first retrieves relevant, up-to-date information from a document collection, then passes that
information to the model so it can generate a more accurate, grounded answer.

Vector Databases store embeddings and allow fast similarity search across millions of documents.
They are a core infrastructure piece behind modern RAG systems, chatbots, and recommendation
engines used by companies around the world today.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_text(article_text)

client = chromadb.Client()
embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.create_collection(name="rag_chunks", embedding_function=embedder)

collection.add(
    documents=chunks,
    metadatas=[{"source": "ai_overview.txt", "chunk_index": i, "char_count": len(c)} for i, c in enumerate(chunks)],
    ids=[f"ai_overview_chunk_{i}" for i in range(len(chunks))],
)
print("Stored original document. Count:", collection.count())

# ---------- Exercise 1 ----------
third_doc_text = """Transformers and Large Language Models

The transformer is a neural network architecture that uses a mechanism called self-attention
to weigh the importance of different words in a sentence. Unlike older recurrent networks,
transformers process all words in parallel, which makes training much faster.

Large Language Models, or LLMs, are transformers trained on huge amounts of text. They can
write essays, answer questions, translate languages, and generate code. Examples include GPT,
Claude, and Llama.
"""

third_doc_chunks = splitter.split_text(third_doc_text)

third_doc_metadatas = [
    {"source": "transformers_and_llms.txt", "chunk_index": i, "char_count": len(c)}
    for i, c in enumerate(third_doc_chunks)
]
third_doc_ids = [f"transformers_and_llms_chunk_{i}" for i in range(len(third_doc_chunks))]

count_before = collection.count()
collection.add(documents=third_doc_chunks, metadatas=third_doc_metadatas, ids=third_doc_ids)
count_after = collection.count()

print("Count before:", count_before)
print("New chunks added:", len(third_doc_chunks))
print("Count after:", count_after)
assert count_after == count_before + len(third_doc_chunks), "Count mismatch!"

all_items = collection.get(include=["metadatas"])
sources = sorted({m["source"] for m in all_items["metadatas"]})
print("\nSources in collection:", sources)
for s in sources:
    n = len(collection.get(where={"source": s})["ids"])
    print(f"  {s}: {n} chunks")