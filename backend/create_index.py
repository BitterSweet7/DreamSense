from sentence_transformers import SentenceTransformer
import pandas as pd
import faiss

# Load data
df = pd.read_csv('dream_data.csv')

# Load embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings
texts = df['Details'].tolist()
embeddings = embedder.encode(texts, convert_to_numpy=True)

# Save FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, 'dream_index.faiss')

print("Index created successfully.")
