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
import numpy as np

# --- Setup FastAPI ---
app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # หรือใส่ URL frontend จริงเช่น ["http://localhost:3000"]
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

def normalize(vecs):
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
# --- Retrieval Function ---
def retrieve_snippet(query, k=3, threshold=0.75):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    query_embedding = normalize(query_embedding)

    distances, indices = index.search(query_embedding, k)
    results = []

    for dist, idx in zip(distances[0], indices[0]):
        similarity = 1 - dist  # ถ้าใช้ FAISS L2 flat → ต้อง normalize ก่อนถึงจะเป็น cosine
        if similarity >= threshold:
            results.append(texts[idx])

    if results:
        return results
    else:
        return ["ขออภัย ฉันไม่พบคำอธิบายที่เกี่ยวข้องในฐานข้อมูลค่ะ 💤"]


# --- Asking Function ---
def ask_query(query: str, character: str = "formal"):
    retrieved_texts = retrieve_snippet(query)

    system_prompt = ""
    
    if character == "formal":
        system_prompt = (
            "You are a warm and emotionally intuitive dream interpreter, unless the user requests your tone otherwise. \n"
            "Your goal is to gently guide the user through an interpretation that feels personal, positive, and meaningful.\n"
            
            "When the user shares a dream: \n"
            "- First try to **sense the emotional tone** of the dream (e.g. sad, anxious, hopeful, lonely, confused, etc). \n"
            "- Then respond using that emotional tone — if the dream is sad or lonely, respond with gentleness and empathy.  \n"
            
            "Here's how you'll respond: \n"
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
        )
    elif character == "Relax":
        system_prompt = (
            "You are a quirky, artistic dream interpreter with an indie vibe. \n"
            "Your goal is to offer a unique, creative interpretation that challenges conventional thinking.\n"
            
            "When the user shares a dream: \n"
            "- Look for the unconventional patterns and unexpected connections in the dream. \n"
            "- Respond with a mix of poetic insights and casual, friendly language. \n"
            
            "Here's how you'll respond: \n"
            "1. Start with a creative, slightly offbeat greeting that matches the dream's energy. \n"
            "2. Interpret the dream using metaphors, cultural references, and artistic perspectives. \n"
            "   ✧ Always highlight key symbols with ***asterisks***. \n"
            "   ✧ Use indie-style markers like (✧, ⋆, ⁂) when describing connections. \n"
            "3. End with an unexpected question that invites the user to think differently about their dream. \n"

            "Style guidelines for your indie personality: \n"
            "- Mix casual slang with occasional poetic phrases. \n"
            "- Use quirky, artistic emojis (✨, 🌈, 🔮, 🎭, 🌿) liberally throughout your response. \n"
            "- Reference indie music, art, or pop culture occasionally if relevant. \n"
            "- Break conventional punctuation and capitalization rules occasionally for effect. \n"
            "- Be warm but with an edge of mysteriousness. \n"
        )
    
    # Add retrieved text to the system prompt
    system_prompt += f"{retrieved_texts[0]}"

    messages = [
        {
            "role": "system",
            "content": system_prompt
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
    character: Optional[str] = "formal"  # default to formal

# --- Route ---
@app.post("/api/analyze")
async def analyze(request: PromptRequest):
    if not request.prompt:
        return {"error": "No prompt provided"}
    
    answer = ask_query(request.prompt, request.character)
    return {"answer": answer}
