from collections import Counter
from rag_pipeline import RAGPipeline

NEW_DOC = """Docker and Containers

Docker packages an application together with everything it needs to run, such as libraries and
settings, into a unit called a container. Because the container carries its own dependencies, the
application behaves the same way on a laptop, a test server, and a production server.

A Docker image is a read-only template used to create containers. A Dockerfile is a text file with
the instructions for building an image. Containers are lighter than virtual machines because they
share the host operating system instead of running a full copy of their own.
"""

rag = RAGPipeline()

before = rag.collection.count()
added = rag.add_document("docker_basics.txt", NEW_DOC)
print(f"\nChunks before: {before} | added: {added} | after: {rag.collection.count()}")

metas = rag.collection.get(include=["metadatas"])["metadatas"]
print("Chunks per source:", dict(Counter(m["source"] for m in metas)))

questions = [
    "What is a Docker container?",
    "How is a Docker image different from a container?",
    "What is the best recipe for chocolate cake?",   
]

for q in questions:
    answer, chunks = rag.ask(q)
    print("\n" + "-" * 70)
    print("Question:", q)
    print("Answer  :", answer)
    if chunks:
        print("Sources :", sorted({f"{c['source']} #{c['chunk_index']}" for c in chunks}))
    else:
        print("Sources : none (no relevant chunks, so the LLM was never called)")