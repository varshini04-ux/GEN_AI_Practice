# performing the safe_add() function using upsert internally

import chromadb

client = chromadb.Client()
collection = client.create_collection(name="safe_add_demo")

def safe_add(collection, id, text):
    """Adds a document if the id is new, or updates it if the id already exists.
    Never raises a duplicate-id error, unlike collection.add()."""
    collection.upsert(
        documents=[text],
        ids=[id]
    )
    print(f"Upserted id='{id}'")

safe_add(collection, "doc1", "Solar panels convert sunlight directly into electricity.")
safe_add(collection, "doc1", "Solar panels convert sunlight into electricity using photovoltaic cells.")
safe_add(collection, "doc2", "Bees play a critical role in pollinating crops worldwide.")

print("\nFinal count:", collection.count())
print("Final state:", collection.get()["ids"])