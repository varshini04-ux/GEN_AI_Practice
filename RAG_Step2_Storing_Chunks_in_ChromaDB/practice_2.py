import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter


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
    metadatas=[
        {"source": "ai_overview.txt", "chunk_index": i, "char_count": len(c), "version": "v1_size300"}
        for i, c in enumerate(chunks)
    ],
    ids=[f"ai_overview_chunk_{i}" for i in range(len(chunks))],
)
small_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=30,
    separators=["\n\n", "\n", ". ", " ", ""],
)
small_chunks = small_splitter.split_text(article_text)

collection.add(
    documents=small_chunks,
    metadatas=[
        {"source": "ai_overview.txt", "chunk_index": i, "char_count": len(c), "version": "v2_size150"}
        for i, c in enumerate(small_chunks)
    ],
    ids=[f"ai_overview_v2_chunk_{i}" for i in range(len(small_chunks))],
)

print("Original chunks (size 300):", len(chunks))
print("New chunks (size 150):", len(small_chunks))
print("Total in collection:", collection.count())

query = "What is Retrieval-Augmented Generation?"


def show(title, version):
    res = collection.query(
        query_texts=[query],
        n_results=2,
        where={"version": version},
        include=["documents", "metadatas", "distances"],
    )
    print(f"\n===== {title} =====")
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        print(f"Distance: {dist:.4f} | Chunk #{meta['chunk_index']} | {meta['char_count']} chars")
        print(doc)
        print("-" * 60)


show("BIG chunks (size 300)", "v1_size300")
show("SMALL chunks (size 150)", "v2_size150")

# Observe:
# - Small chunks are more precise but may lose surrounding context.
# - Big chunks carry more context but can include unrelated sentences.
