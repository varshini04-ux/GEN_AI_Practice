import chromadb

client = chromadb.Client()
collection = client.create_collection(name="delete_by_category_demo")

collection.add(
    documents=[
        "Solar panels convert sunlight directly into electricity.",
        "Yoga combines physical postures with breathing exercises.",
        "The stock exchange opens at 9:15 AM on weekdays in India.",
        "Bees play a critical role in pollinating crops worldwide.",
        "Wind turbines generate power by capturing kinetic energy."
    ],
    metadatas=[
        {"category": "energy"},
        {"category": "fitness"},
        {"category": "finance"},
        {"category": "nature"},
        {"category": "energy"}
    ],
    ids=["doc1", "doc2", "doc3", "doc4", "doc5"]
)

print("Count before delete:", collection.count())

collection.delete(where={"category": "energy"})

print("Count after delete:", collection.count())
print("Remaining ids:", collection.get()["ids"])