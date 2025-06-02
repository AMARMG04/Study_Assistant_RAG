# ingest.py
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# --- Configuration ---
DATA_PATH = "data" # Directory where your study documents are
CHROMA_PATH = "chroma_db" # Directory to store your ChromaDB
OLLAMA_EMBED_MODEL = "nomic-embed-text:v1.5"

def ingest_documents():
    print(f"Loading documents from {DATA_PATH}...")
    documents = []

    # Use DirectoryLoader to load various file types
    # You might need to install 'unstructured' if you have .docx, .epub etc.
    # pip install unstructured # and its dependencies, potentially problematic on M1
    # For simplicity, let's stick to PDF and TXT for now.
    for root, _, files in os.walk(DATA_PATH):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file.endswith(".txt") or file.endswith(".md"):
                loader = TextLoader(file_path)
            # Add more loaders as needed for other file types
            else:
                print(f"Skipping unsupported file type: {file_path}")
                continue
            documents.extend(loader.load())

    if not documents:
        print("No documents found to process. Please place your study materials in the 'data' folder.")
        return

    print(f"Loaded {len(documents)} documents. Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # Adjust based on your content and LLM context window
        chunk_overlap=200, # Overlap to maintain context between chunks
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    print(f"Creating embeddings with {OLLAMA_EMBED_MODEL} and storing in ChromaDB...")
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)

    # Create a new ChromaDB from the chunks and embeddings
    # If the directory exists, it will re-create/overwrite the existing data
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"ChromaDB created and persisted at {CHROMA_PATH}")

if __name__ == "__main__":
    ingest_documents()