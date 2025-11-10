from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .rag import get_answer  # Make sure rag.py is in the same folder

app = FastAPI()

# ✅ CORS configuration for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Use ["*"] for testing, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Request schema
class QueryRequest(BaseModel):
    query: str

# ✅ POST endpoint
@app.post("/api/query")
async def query_endpoint(request: QueryRequest):
    answer = get_answer(request.query)
    return {
        "query": request.query,
        "answer": answer
    }