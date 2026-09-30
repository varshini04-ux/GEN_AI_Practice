# One-Hot Encoding with your own words
from sklearn.preprocessing import OneHotEncoder

words = [['sun'], ['moon'], ['ocean'], ['mountain'], ['sun'], ['ocean']]

encoder = OneHotEncoder(sparse_output=False)
result = encoder.fit_transform(words)

print("words :", encoder.categories_[0])
print("Embeddings:")
print(result)
