import pandas as pd
import faiss
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

# --- Setup FastAPI ---
app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ใส่ URL frontend จริงเช่น ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load LLaMA model ---
checkpoint = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
llama_model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16)
device = 0 if torch.cuda.is_available() else -1

generator = pipeline(
    "text-generation",
    model=llama_model,
    tokenizer=tokenizer,
    device=device
)

# --- Load FAISS + Embedder ---
dream_data = pd.read_csv("dream_data.csv")
texts = dream_data["Details"].tolist()
terms = dream_data["Term"].tolist()
index = faiss.read_index('dream_index.faiss')
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# --- Retrieval Function ---
def retrieve_snippet(query, k=1):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    return [texts[i] for i in indices[0]]

# --- Asking Function ---
def ask_query(query: str):
    retrieved_texts = retrieve_snippet(query)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a warm and emotionally intuitive dream interpreter, unless the user requests your tone otherwise. \n"
                "Your goal is to gently guide the user through an interpretation that feels personal, positive, and meaningful.\n"
                
                "When the user shares a dream: \n"
                "- First try to **sense the emotional tone** of the dream (e.g. sad, anxious, hopeful, lonely, confused, etc). \n"
                "- Then respond using that emotional tone — if the dream is sad or lonely, respond with gentleness and empathy.  \n"
                
                "Here’s how you’ll respond: \n"
                "1. Begin with a friendly sentence and use emojis to match the dream's mood. \n"
                "2. Break down the dream using clear interpretations of ***symbols*** and ***feelings***. \n"
                "   ➤ Always cover the main words with ***word***. \n"
                "   ➤ Use soft bullet points (– or ➤) when describing the meaning of each symbol. \n"
                "3. End with a short reflective takeaway and 1 meaningful question for the user to consider. \n"

                " Here are some gentle formatting suggestions to keep the response clear and human-like: \n"
                "- Avoid complex or technical language. \n"
                "- Use soft, nurturing tone — like a caring friend explaining the dream. \n"
                "- Highlight important **symbols**, **feelings**, and **themes** using triple asterisks like this: `***loneliness***`, `***freedom***`. \n"
                "- Add emojis throughout (🌙, 💭, 🌌, 🐍, 🦋) when they relate to specific dream elements or emotions. \n"
                "- Use line breaks or bullets to make your response more readable and friendly. \n"

                f"{retrieved_texts[0]}"
            )
        },
        {
            "role": "user",
            "content": query
        }
    ]

    response = generator(messages, max_new_tokens=512)[-1]["generated_text"][-1]["content"]
    return response

# --- FastAPI Input Schema ---
class PromptRequest(BaseModel):
    prompt: str
    language: Optional[str] = "en"

# --- Route ---
@app.post("/api/analyze")
async def analyze(request: PromptRequest):
    if not request.prompt:
        return {"error": "No prompt provided"}
    
    answer = ask_query(request.prompt)
    return {"answer": answer}
