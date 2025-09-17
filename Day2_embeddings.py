# day2_embeddings.py
# Day 2: Understanding meaning with embeddings + similarity search

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1) Load a small pre-trained embeddings model
model = SentenceTransformer("all-MiniLM-L6-v2")  # light, fast, good quality

# 2) Our "knowledge base" (like sentences in a library)
sentences = [
    "Soccer is played with a round ball.",
    "Pizza is a popular Italian dish.",
    "Basketball is played by shooting into a hoop.",
    "The sun is the center of our solar system.",
    "Dogs are loyal animals."
]

# 3) Encode each sentence into embeddings (numbers that represent meaning)
embeddings = model.encode(sentences)

# 4) Ask a question
query = "mY NAME IS MEENA"
query_embedding = model.encode([query])

# 5) Find similarity between query and each sentence
similarities = cosine_similarity(query_embedding, embeddings)[0]

# 6) Pick the most similar sentence
best_match_index = similarities.argmax()
best_match = sentences[best_match_index]

print("\n=== Question ===")
print(query)
print("\n=== Closest Match ===")
print(best_match)
# ...existing code...

# # 3) Encode each sentence into embeddings (numbers that represent meaning)
# embeddings = model.encode(sentences)

# # Print embeddings for each sentence
# print("\n=== Sentence Embeddings ===")
# for sentence, embedding in zip(sentences, embeddings):
#     print(f"Sentence: {sentence}")
#     print(f"Embedding: {embedding}\n")

