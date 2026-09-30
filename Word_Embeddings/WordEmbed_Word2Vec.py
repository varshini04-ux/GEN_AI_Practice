#Word2Vec with a bigger custom corpus

from gensim.models import Word2Vec

sentences = [
    ["football", "is", "a", "team", "sport"],
    ["cricket", "is", "a", "team", "sport"],
    ["football", "is", "popular", "worldwide"],
    ["cricket", "is", "popular", "in", "India"],
    ["football", "requires", "stamina"],
    ["cricket", "requires", "strategy"],
    ["chess", "is", "a", "strategy", "game"],
    ["chess", "requires", "patience"],
]

model = Word2Vec(sentences, vector_size=10, window=2, min_count=1, workers=1)

def check_similarity(word1, word2):
    score = model.wv.similarity(word1, word2)
    print(f"Similarity between '{word1}' and '{word2}': {score:.3f}")

check_similarity('football', 'cricket')
check_similarity('cricket', 'chess')
check_similarity('football', 'chess')