# metadata categories + filtering across two categories

import chromadb

client = chromadb.Client()
collection = client.create_collection(name="categorized_collection")

collection.add(
    documents=[
        "Solar panels convert sunlight directly into electricity.",
        "Wind turbines generate power by capturing kinetic energy from wind.",
        "Yoga combines physical postures with breathing exercises.",
        "Strength training builds muscle through resistance exercises.",
        "The stock exchange opens at 9:15 AM on weekdays in India.",
        "Mutual funds pool money from investors to buy diversified assets.",
        "Bees play a critical role in pollinating crops worldwide.",
        "Coral reefs are home to thousands of marine species."
    ],
    metadatas=[
        {"category": "energy"},
        {"category": "energy"},
        {"category": "fitness"},
        {"category": "fitness"},
        {"category": "finance"},
        {"category": "finance"},
        {"category": "nature"},
        {"category": "nature"}
    ],
    ids=["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8"]
)

print("Documents added! Total documents:", collection.count())

print("\n Query filtered to 'energy' ")
results_energy = collection.query(
    query_texts=["How do we generate power sustainably?"],
    n_results=2,
    where={"category": "energy"}
)
for doc, distance in zip(results_energy["documents"][0], results_energy["distances"][0]):
    print(f"Distance: {distance:.4f}  |  {doc}")

print("\n--- Query filtered to 'finance' ---")
results_finance = collection.query(
    query_texts=["How should I invest my money?"],
    n_results=2,
    where={"category": "finance"}
)
for doc, distance in zip(results_finance["documents"][0], results_finance["distances"][0]):
    print(f"Distance: {distance:.4f}  |  {doc}")
