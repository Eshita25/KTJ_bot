# embedding.py
from langchain.embeddings.base import Embeddings
import google.generativeai as genai

class GoogleGenAIEmbeddings(Embeddings):
    def __init__(self, api_key):
        self.api_key = api_key
        genai.configure(api_key=api_key)

    def embed_documents(self, texts):
        return [
            genai.embed_content(
                model='models/embedding-001',
                content=text,
                task_type="retrieval_document"
            )['embedding']
            for text in texts
        ]

    def embed_query(self, text):
        return genai.embed_content(
            model='models/embedding-001',
            content=text,
            task_type="retrieval_query"
        )['embedding']
