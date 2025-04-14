import pandas as pd
import faiss
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


# Load LLaMA model
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

# Load the CSV file
dream_data = pd.read_csv("dream_data.csv")
texts = dream_data["Details"].tolist()
terms = dream_data["Term"].tolist()

# Load FAISS index
index = faiss.read_index('dream_index.faiss')

# Load embedder (only to encode the query)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Retrieval function (ใช้ FAISS)
def retrieve_snippet(query, k=1):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    retrieved_texts = [texts[i] for i in indices[0]]
    return retrieved_texts

# Asking function
def ask_query(query):
    retrieved_texts = retrieve_snippet(query)

    # Prepare the messages for the text generation pipeline
    messages = [
    {
        "role": "system",
        "content": (
            "You are a professional Dream Analyst AI.\n"
            "Your task is to analyze the dream based on the context provided.\n"
            "Format your answer as:\n"
            "1. Start with identifying the **overall emotional tone** of the dream in 1-2 sentences and relevant emoji.\n"
            "2. Then, provide **bullet points** (➤) listing key symbols or meanings from the dream, each symbol with a brief description and relevant emoji.\n"
            "3. Keep the explanation concise and easy to understand.\n"
            "4. Do not add any information outside the context given.\n\n"
            "Context:\n"
            f"{retrieved_texts[0]}"
        )
    },
    {
        "role": "user",
        "content": query
    }
]


    # Generate a response using the text generation pipeline
    response = generator(messages, max_new_tokens=256)[-1]["generated_text"][-1]["content"]
    # print(f"Query: \n\t{query}")
    # print(f"Context: \n\t{retrieved_texts[0]}")
    # print(f"Answer: \n\t{response}")
    return response

# Example query
# query = "I dreamt that I was getting dragged to the red clothe covering round table, it looked like tables from those wedding event. With hairs coming out under it grabbing my leg dragged me inside, I felt terrified, woke up in cold sweats. What does that mean?"
# ask_query(query)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    prompt = data.get('prompt')

    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400

    answer = ask_query(prompt)
    return jsonify({'answer': answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
