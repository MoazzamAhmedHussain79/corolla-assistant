import os
import fitz  # PyMuPDF
import faiss
import numpy as np
import httpx
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv(dotenv_path="api/.env")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
print("🔑 API key loaded:", os.getenv("OPENROUTER_API_KEY"))
if not OPENROUTER_API_KEY:
    raise EnvironmentError("❌ OPENROUTER_API_KEY not found in .env file.")

# Initialize OpenAI client for chat completion
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

PDF_PATH = r"C:\Users\Moazam Ahmed\Downloads\Corolla_E11_Haynes_Workshop_Manual.pdf"
INDEX_PATH = "api/corolla_index.faiss"
CHUNK_SIZE = 500
EMBED_MODEL = "openai/text-embedding-ada-002"  # Valid embedding model on OpenRouter

# Step 1: Load and chunk PDF
def load_pdf_chunks(path):
    doc = fitz.open(path)
    chunks = []
    for page in doc:
        text = (page.get_text() or "").strip()
        for i in range(0, len(text), CHUNK_SIZE):
            chunk = text[i:i+CHUNK_SIZE]
            if chunk:
                chunks.append(chunk)
    return chunks

# Step 2: Embed text using OpenRouter (httpx)
def embed_text(texts):
    embeddings = []
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    for i in tqdm(range(0, len(texts), 10), desc="Embedding chunks"):
        batch = texts[i:i+10]
        payload = {
            "model": EMBED_MODEL,
            "input": batch
        }
        response = httpx.post("https://openrouter.ai/api/v1/embeddings", json=payload, headers=headers)
        data = response.json()
        if "data" not in data:
            print("❌ Embedding failed:", data)
            raise ValueError(f"Embedding error: {data}")
        batch_embeddings = [e["embedding"] for e in data["data"]]
        embeddings.extend(batch_embeddings)
    return np.array(embeddings, dtype="float32")

# Step 3: Build FAISS index
def build_index():
    chunks = load_pdf_chunks(PDF_PATH)
    print(f"📄 Loaded {len(chunks)} chunks from PDF.")
    embeddings = embed_text(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)
    with open("api/chunks.txt", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk + "\n")
    print("✅ FAISS index saved.")

# Step 4: Chat completion using OpenRouter (OpenAI SDK)
def chat_completion(prompt: str) -> str:
    completion = client.chat.completions.create(
        model="openrouter/polaris-alpha",
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content

# Step 5: Query FAISS index
def get_answer(query: str) -> str:
    if not os.path.exists(INDEX_PATH):
        return "Index not found. Please run build_index() first."

    index = faiss.read_index(INDEX_PATH)
    with open("api/chunks.txt", "r", encoding="utf-8") as f:
        chunks = f.readlines()

    query_embedding = embed_text([query])
    D, I = index.search(query_embedding, k=3)
    retrieved = [chunks[i].strip() for i in I[0]]

    prompt = "Answer the question based on the following context:\n\n"
    prompt += "\n---\n".join(retrieved)
    prompt += f"\n\nQuestion: {query}\nAnswer:"

    return chat_completion(prompt)

# Optional: Build index when run directly
if __name__ == "__main__":
    build_index()