from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sentence_transformers import SentenceTransformer
import faiss
import json
from pathlib import Path

# ====================== LOAD FAQs ======================
faqs_path = Path("all.json")

with open(faqs_path, "r", encoding="utf-8") as f:
    faqs = json.load(f)

print(f"Loaded {len(faqs)} FAQs")

embedder = SentenceTransformer('all-MiniLM-L6-v2')
questions = [item["question"] for item in faqs]
answers = [item["answer"] for item in faqs]

question_embeddings = embedder.encode(questions, convert_to_numpy=True)

dimension = question_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(question_embeddings)

print("Search index ready!")

# ====================== CREATE API ======================
app = FastAPI(title="Indian Banking FAQ Bot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ask")
def ask_question(q: str = Query(..., description="Your question")):
    query_vec = embedder.encode([q], convert_to_numpy=True)
    distances, indices = index.search(query_vec, 3)

    results = []
    for i, idx in enumerate(indices[0]):
        score = float(1 / (1 + distances[0][i]))
        results.append({
            "matched_question": questions[idx],
            "answer": answers[idx],
            "score": round(score, 3)
        })

    return {
        "question": q,
        "matches": results
    }

# Serve the frontend
@app.get("/")
def serve_frontend():
    return FileResponse("index.html")