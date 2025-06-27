# build_index.py
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from dotenv import load_dotenv 
from embedding import GoogleGenAIEmbeddings


PDF_PATH = "use.pdf"
INDEX_PATH = "faiss_index_store"

# Load PDF
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

# Split text into chunks
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
split_documents = splitter.split_documents(documents)

# Embed and create FAISS index
load_dotenv()
GOOGLE_API_KEY =config("GOOGLE_API_KEY")
embedding_model = GoogleGenAIEmbeddings(api_key=GOOGLE_API_KEY)
vectorstore = FAISS.from_documents(split_documents, embedding_model)
vectorstore.save_local(INDEX_PATH)

print("✅ Index built and saved at:", INDEX_PATH)
