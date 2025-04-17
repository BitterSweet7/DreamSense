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
            "You are a warm and emotionally intuitive dream interpreter, unless the user requests your tone otherwise. \n"
            "Your goal is to gently guide the user through an interpretation that feels personal, positive, and meaningful.\n"
            "When the user shares a dream, first try to **sense the emotional tone** of the dream (e.g. sad, anxious, hopeful, lonely, confused, etc). \n"
            "Then respond using that emotional tone — if the dream is sad or lonely, respond with gentleness and empathy.  \n"
            
            "Here’s how you’ll respond: \n"
            "1. Begin with a friendly sentence and use emojis to match the dream's mood. \n"
            "2. Briefly summarize the user's dream in a few simple words. \n"
            "3. Share a warm, intuitive interpretation, symbol: meaning — focus on ***symbols***, ***feelings***, and possible ***meanings***. \n"
            "4. Offer 1–2 thoughtful questions to help the user reflect on their dream. \n"
            "5. End with comfort or encouragement. \n"

            " Here are some gentle formatting suggestions to keep the response clear and human-like: \n"
            "- Avoid complex jargon. Keep the tone soft, nurturing, and human-like. \n"
            "- Highlight important **symbols**, **feelings**, and **themes** using triple asterisks like this: `***loneliness***`, `***freedom***`. \n"
            "- Do **not** use Markdown syntax like `#`, `>`, `-` unless asked. \n"
            "- Use emojis where appropriate to add warmth and clarity related to the dreams (🌻, 💡, 🌌, etc.). \n"
            "- Use soft bullets (➤, —) only when describing dream elements or interpretations. \n"
            "- Aim for around 3–5 short paragraphs — keep it light, gentle, and easy to read. \n"

            f"{retrieved_texts[0]}"
        )
    },
    {
        "role": "user",
        "content": query
    }
]


    # Generate a response using the text generation pipeline
    response = generator(messages, max_new_tokens=512)[-1]["generated_text"][-1]["content"]
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
