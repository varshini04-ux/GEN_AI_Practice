# my own 5 sentences in a new collection

import chromadb

client = chromadb.Client()
collection = client.create_collection(name="my_own_collection")

collection.add(
    documents=[
        "Solar panels convert sunlight directly into electricity.",
        "A strong password should mix letters, numbers, and symbols.",
        "Yoga combines physical postures with breathing exercises.",
        "The stock exchange opens at 9:15 AM on weekdays in India.",
        "Bees play a critical role in pollinating crops worldwide."
    ],
    ids=["s1", "s2", "s3", "s4", "s5"]
)

print("Documents added! Total documents:", collection.count())

results = collection.query(
    query_texts=["How can I generate clean energy?"],
    n_results=2
)

for doc, distance in zip(results["documents"][0], results["distances"][0]):
    print(f"Distance: {distance:.4f}  |  {doc}")
