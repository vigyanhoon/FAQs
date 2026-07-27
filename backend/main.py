from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastembed import TextEmbedding
import faiss
import numpy as np
import json
from pathlib import Path

# ====================== LOAD FAQs ======================
faqs_path = Path("all.json")

with open(faqs_path, "r", encoding="utf-8") as f:
    faqs = json.load(f)

print(f"Loaded {len(faqs)} FAQs")

questions = [item["question"] for item in faqs]
answers = [item["answer"] for item in faqs]

# ====================== LIGHTWEIGHT EMBEDDING MODEL ======================
print("Loading lightweight embedding model...")
embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")  # very light & good quality

print("Creating embeddings...")
question_embeddings = np.array(list(embedder.embed(questions)))

# ====================== BUILD FAISS INDEX ======================
dimension = question_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(question_embeddings.astype(np.float32))

print("Search index ready!")

# ====================== FASTAPI APP ======================
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
    query_embedding = np.array(list(embedder.embed([q]))).astype(np.float32)
    
    distances, indices = index.search(query_embedding, 3)

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

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")