from rag_pipeline import RAGPipeline

rag = RAGPipeline()

question = "How do I keep my API key secret when I push my project to GitHub?"


REQUIRED = {
    ".env file": [".env"],
    ".gitignore": [".gitignore"],
}

summary = []
for n in [1, 2, 4]:
    answer, chunks = rag.ask(question, n_results=n)
    covered = [name for name, keys in REQUIRED.items() if any(k in answer.lower() for k in keys)]
    summary.append((n, len(chunks), covered))

    print("\n" + "=" * 70)
    print(f"n_results = {n} -> {len(chunks)} chunk(s) used")
    for c in chunks:
        print(f"   [{c['similarity']}] {c['source']} #{c['chunk_index']}: {c['text'][:60]}...")
    print("\nAnswer:\n" + answer)

print("\n" + "=" * 70)
print(f"{'n_results':<12}{'chunks':<10}covered")
for n, used, covered in summary:
    print(f"{n:<12}{used:<10}{', '.join(covered) or 'nothing'} ({len(covered)}/{len(REQUIRED)})")