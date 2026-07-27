from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from pathlib import Path

# ====================== LOAD FAQs ======================
faqs_path = Path("/Users/sanjaypandey/Desktop/projects/faqs/all.json")

with open(faqs_path, "r", encoding="utf-8") as f:
    faqs = json.load(f)

print(f"Loaded {len(faqs)} FAQs from {faqs_path.name}")

# ====================== CREATE EMBEDDINGS ======================
print("Loading embedding model (first time it will download ~80MB)...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

questions = [item["question"] for item in faqs]
answers   = [item["answer"] for item in faqs]

print("Creating embeddings...")
question_embeddings = embedder.encode(questions, convert_to_numpy=True, show_progress_bar=True)

# ====================== BUILD SEARCH INDEX ======================
dimension = question_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(question_embeddings)

print(f"\nIndex ready with {len(faqs)} FAQs\n")

# ====================== ASK FUNCTION ======================
def ask(query, top_k=2):
    query_vec = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        score = float(1 / (1 + distances[0][i]))
        results.append({
            "matched_question": questions[idx],
            "answer": answers[idx],
            "score": score
        })
    return results

# ====================== INTERACTIVE CHAT ======================
print("Banking FAQ Bot is ready!")
print("Type your question (or type 'exit' to quit)\n")

while True:
    query = input("You: ").strip()
    if query.lower() in ["exit", "quit", "q"]:
        break
    if not query:
        continue

    results = ask(query)

    print("\n" + "─" * 60)
    for i, res in enumerate(results, 1):
        print(f"\nMatch {i}  (score: {res['score']:.3f})")
        print(f"Q: {res['matched_question']}")
        print(f"A: {res['answer']}")
    print("─" * 60 + "\n")