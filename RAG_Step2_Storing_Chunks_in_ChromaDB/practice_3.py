

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter


def add_document(collection, splitter, text, source_name, id_prefix=None):
    """Chunk a text, build metadata + ids, and store it in the collection."""
    id_prefix = id_prefix or source_name.rsplit(".", 1)[0]

    doc_chunks = splitter.split_text(text)

    metadatas = [
        {"source": source_name, "chunk_index": i, "char_count": len(c)}
        for i, c in enumerate(doc_chunks)
    ]
    ids = [f"{id_prefix}_chunk_{i}" for i in range(len(doc_chunks))]

    collection.add(documents=doc_chunks, metadatas=metadatas, ids=ids)

    print(f"Added {len(doc_chunks)} chunks from '{source_name}'. Total now: {collection.count()}")
    return len(doc_chunks)



splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
)

client = chromadb.Client()
embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.create_collection(name="rag_chunks", embedding_function=embedder)


doc1 = """Machine Learning is a subset of AI where systems learn patterns from data instead of following
hardcoded rules. This approach powers everything from spam filters to recommendation systems.

Deep Learning uses neural networks with many layers to discover complex patterns in large data.
"""

doc2 = """Cloud Computing Basics

Cloud computing lets people rent computing power, storage, and software over the internet
instead of buying and maintaining their own hardware. Major providers include AWS, Azure,
and Google Cloud.

Companies use the cloud to scale quickly, pay only for what they use, and deploy applications
around the world in minutes.
"""

add_document(collection, splitter, doc1, "ml_basics.txt")
add_document(collection, splitter, doc2, "cloud_basics.txt")


results = collection.query(
    query_texts=["why do companies use the cloud"],
    n_results=2,
    include=["documents", "metadatas", "distances"],
)
for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
    print(f"Distance: {dist:.4f} | Source: {meta['source']} | Chunk #{meta['chunk_index']}")
    print(doc)
    print("-" * 60)