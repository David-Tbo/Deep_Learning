from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

corpus = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are friends",
]

# Bag of Words: raw token counts, no notion of importance
bow = CountVectorizer()
bow_matrix = bow.fit_transform(corpus)
print(bow.get_feature_names_out())
print(bow_matrix.toarray())