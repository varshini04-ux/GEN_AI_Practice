# MiniLM vs MPNet comparison



import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')
model_mpnet = SentenceTransformer('all-mpnet-base-v2')
print("Both models loaded successfully!")

my_documents = [
    "The International Space Station orbits Earth roughly every 90 minutes.",
    "A good curry balances spice, acidity, and richness in every bite.",
    "Compound interest allows savings to grow faster over long time periods.",
    "Elephants are highly social animals that live in matriarchal herds.",
    "The Renaissance was a period of major cultural and artistic rebirth in Europe.",
    "Electric vehicles rely on lithium-ion batteries for their power storage.",
    "Coral reefs support roughly a quarter of all marine species on Earth.",
    "A well-diversified portfolio spreads risk across different asset classes.",
    "The human heart beats around 100,000 times every single day.",
    "Ancient Rome built an extensive network of roads connecting its empire."
]

start = time.time()
minilm_embeddings = model.encode(my_documents)
minilm_time = time.time() - start

start = time.time()
mpnet_embeddings = model_mpnet.encode(my_documents)
mpnet_time = time.time() - start

print(f"MiniLM embedding shape: {minilm_embeddings.shape}  |  Time: {minilm_time:.3f}s")
print(f"MPNet  embedding shape: {mpnet_embeddings.shape}  |  Time: {mpnet_time:.3f}s")

def compare_models(query, docs, top_k=3):
    print(f"Query: {query}\n")

    q_mini = model.encode([query])
    scores_mini = cosine_similarity(q_mini, minilm_embeddings)[0]
    ranked_mini = scores_mini.argsort()[::-1][:top_k]

    q_mpnet = model_mpnet.encode([query])
    scores_mpnet = cosine_similarity(q_mpnet, mpnet_embeddings)[0]
    ranked_mpnet = scores_mpnet.argsort()[::-1][:top_k]

    print("MiniLM (fast, 384-dim):")
    for idx in ranked_mini:
        print(f"  {scores_mini[idx]:.3f}  |  {docs[idx]}")

    print("\nMPNet (accurate, 768-dim):")
    for idx in ranked_mpnet:
        print(f"  {scores_mpnet[idx]:.3f}  |  {docs[idx]}")
    print()

compare_models("How does the body work?", my_documents)
compare_models("What lives in the ocean?", my_documents)