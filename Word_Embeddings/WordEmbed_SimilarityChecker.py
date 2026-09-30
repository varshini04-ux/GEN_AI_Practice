# Build a word similarity checker function


from gensim.models import Word2Vec

sentences = [
    ["coffee", "is", "a", "popular", "morning", "drink"],
    ["tea", "is", "a", "popular", "evening", "drink"],
    ["coffee", "gives", "energy"],
    ["tea", "is", "calming"],
    ["coffee", "is", "bitter"],
    ["tea", "is", "soothing"],
    ["coffee", "and", "tea", "are", "beverages"],
]

model = Word2Vec(sentences, vector_size=10, window=2, min_count=1, workers=1)

print(model.wv.most_similar('coffee', topn=3))
print(model.wv.most_similar('tea', topn=3))