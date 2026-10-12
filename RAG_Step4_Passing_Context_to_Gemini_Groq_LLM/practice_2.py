from difflib import SequenceMatcher
from itertools import combinations
from rag_pipeline import RAGPipeline

rag = RAGPipeline()

question = "Why should each Python project use its own virtual environment?"
chunks = rag.retrieve(question)
prompt = rag.build_prompt(question, chunks)

RUNS = 3


def consistency(answers):
    """Average text similarity between every pair of answers (1.0 = identical)."""
    pairs = list(combinations(answers, 2))
    return sum(SequenceMatcher(None, a, b).ratio() for a, b in pairs) / len(pairs)


results = {}
for temp in [0.1, 1.0]:
    answers = [rag.llm.generate(prompt, rag.system_prompt, temperature=temp) for _ in range(RUNS)]
    results[temp] = consistency(answers)

    print("\n" + "=" * 70)
    print(f"temperature = {temp}")
    for i, a in enumerate(answers, 1):
        print(f"\nRun {i}:\n{a}")

print("\n" + "=" * 70)
print("CONSISTENCY SCORE (1.0 = identical answers every run)")
for temp, score in results.items():
    print(f"  temperature {temp}: {score:.2f}")