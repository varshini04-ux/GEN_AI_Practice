import re
from rag_pipeline import RAGPipeline, NO_ANSWER

BULLET_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the provided context. "
    f"If the answer is not in the context, reply exactly: \"{NO_ANSWER}\" "
    "Format rules: respond with EXACTLY 3 bullet points, each starting with '- ' and each one short "
    "sentence. Write nothing before or after the bullets, except a final line 'Sources: <file names>'."
)

rag = RAGPipeline()


def count_bullets(text):
    return len(re.findall(r"^\s*-\s+", text, flags=re.MULTILINE))


def three_bullets(question, max_tries=3):
    chunks = rag.retrieve(question)
    if not chunks:
        return NO_ANSWER, 0, 0

    base_prompt = rag.build_prompt(question, chunks)
    prompt = base_prompt
    for attempt in range(1, max_tries + 1):
        text = rag.llm.generate(prompt, BULLET_PROMPT, temperature=0.1)
        n = count_bullets(text)
        if n == 3 or text.lower().startswith("i don"):
            return text, n, attempt
        prompt = base_prompt + f"\n\nYour previous reply had {n} bullet points. Reply again with EXACTLY 3 bullet points."
    return text, n, max_tries


questions = [
    "What does a Git commit store?",
    "How do file permissions and the chmod command work in Linux?",
    "What is the best recipe for chocolate cake?",
]

for q in questions:
    text, n, attempts = three_bullets(q)
    print("\n" + "-" * 70)
    print("Question:", q)
    print(text)
    print(f"[bullets: {n} | attempts: {attempts}]")