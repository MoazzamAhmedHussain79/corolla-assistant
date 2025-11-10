from rag import get_answer

query = "How do I change the oil?"
answer = get_answer(query)
print("\n🛠️ Answer:\n", answer)