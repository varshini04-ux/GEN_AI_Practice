# FAQ bot


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully!")

faq = [
    ("What is a black hole?", "A black hole is a region of space where gravity is so strong that nothing, not even light, can escape."),
    ("How do vaccines work?", "Vaccines train the immune system to recognize and fight a specific pathogen without causing the disease itself."),
    ("What is inflation?", "Inflation is the rate at which the general price level of goods and services rises over time."),
    ("How does photosynthesis work?", "Photosynthesis is the process plants use to convert sunlight, water, and carbon dioxide into energy and oxygen."),
    ("What is a blockchain?", "A blockchain is a distributed digital ledger that records transactions across many computers securely."),
]

faq_questions = [q for q, a in faq]
faq_answers = [a for q, a in faq]

faq_embeddings = model.encode(faq_questions)

def faq_bot(user_question, threshold=0.45):
    query_embedding = model.encode([user_question])
    scores = cosine_similarity(query_embedding, faq_embeddings)[0]

    best_idx = scores.argmax()
    best_score = scores[best_idx]

    print(f"You asked: {user_question}")
    if best_score < threshold:
        print(f"Bot: Sorry, I don't have an answer for that. (best match score: {best_score:.3f})")
    else:
        print(f"Bot: {faq_answers[best_idx]}")
        print(f"(matched: '{faq_questions[best_idx]}', score: {best_score:.3f})")
    print()

faq_bot("Why can't light escape a black hole?")
faq_bot("How does my immune system learn to fight disease?")
faq_bot("Why do prices go up over time?")
faq_bot("What's the weather like today?")