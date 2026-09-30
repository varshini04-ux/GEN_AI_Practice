

import chromadb

client = chromadb.Client()
collection = client.create_collection(name="upsert_demo")

collection.add(
    documents=[
        "Solar panels convert sunlight directly into electricity.",
        "Yoga combines physical postures with breathing exercises.",
        "The stock exchange opens at 9:15 AM on weekdays in India.",
        "Bees play a critical role in pollinating crops worldwide.",
        "Wind turbines generate power by capturing kinetic energy."
    ],
    ids=["doc1", "doc2", "doc3", "doc4", "doc5"]
)

print("After initial add, count:", collection.count())

collection.upsert(
    documents=[
        "Solar panels convert sunlight into electricity using photovoltaic cells.",
        "The stock exchange opens at 9:15 AM and closes at 3:30 PM on weekdays in India.",
        "Rainforests host more than half of the world's plant and animal species."
    ],
    ids=["doc1", "doc3", "doc6"]
)

print("After upsert, count:", collection.count())

result = collection.get(ids=["doc1", "doc3", "doc6"])
for doc_id, doc in zip(result["ids"], result["documents"]):
    print(f"{doc_id} -> {doc}")