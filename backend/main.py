# main.py
from fastapi import FastAPI, Query
from langchain.vectorstores import FAISS
from embedding import GoogleGenAIEmbeddings
import google.generativeai as genai
import os

from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

app = FastAPI()
load_dotenv()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GOOGLE_API_KEY=config("GOOGLE_API_KEY")

INDEX_PATH = "faiss_index_store"

# Setup embedding and load index
embedding_model = GoogleGenAIEmbeddings(api_key=GOOGLE_API_KEY)
vectorstore = FAISS.load_local(INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

@app.get("/query")
def query_docs(q: str = Query(...)):
    results = vectorstore.similarity_search(q, k=3)

    # If no relevant chunks are found, send a helpful fallback message
    if not results:
        return {
            "answer": "I'm a KTJ bot and I can help with questions related to KTJ. Please visit https://ktj.in for more information."
        }

    # Join top matching chunks
    context = "\n\n".join([doc.page_content for doc in results])

    # Prompt for Gemini
    prompt = f"""
You are a helpful and polite bot that answers user questions based on information about KTJ competitions.
If the answer is not available in the given context, do not make things up — instead say: "I'm a KTJ bot and can answer only KTJ-related queries. Please visit https://ktj.in for more info."

Context:
{context}

Question:
{q}
"""

    response = model.generate_content(prompt)
    return {"answer": response.text.strip()}
