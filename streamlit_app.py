import streamlit as st
import os
import shutil

# --- NEW IMPORTS ---
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma # Updated import for Chroma
# --- END NEW IMPORTS ---

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Configuration ---
UPLOAD_DIR = "uploaded_data"
CHROMA_PATH = "chroma_db"
OLLAMA_LLM_MODEL = "gemma3:12b"
OLLAMA_EMBED_MODEL = "nomic-embed-text:v1.5"

# --- Helper Functions ---

# Function to clear and re-initialize ChromaDB and uploaded data
def reset_application_state():
    """Removes existing ChromaDB and uploaded files, then re-initializes."""
    st.session_state.messages = []
    st.session_state.chroma_ready = False
    st.session_state.rag_chain = None
    st.session_state.vectorstore = None # Ensure this is also reset
    
    if os.path.exists(CHROMA_PATH):
        st.write("Clearing existing ChromaDB...")
        shutil.rmtree(CHROMA_PATH)
    if os.path.exists(UPLOAD_DIR):
        st.write("Clearing uploaded data directory...")
        shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    # --- IMPORTANT FIX: Clear Streamlit's resource cache ---
    # This forces @st.cache_resource functions like load_rag_chain to re-run
    st.cache_resource.clear() 
    
    st.success("Application state reset. Please upload new documents.")

# Function to ingest documents and build/update ChromaDB
# This function no longer returns the vectorstore object directly.
# It ensures the database is created/updated on disk.
def ingest_documents_to_chromadb(uploaded_files):
    if not uploaded_files:
        st.error("No files selected for processing.")
        return False # Indicate failure

    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    st.info("Processing uploaded documents...")
    documents = []
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if uploaded_file.name.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif uploaded_file.name.lower().endswith(".txt") or uploaded_file.name.lower().endswith(".md"):
            loader = TextLoader(file_path)
        else:
            st.warning(f"Skipping unsupported file type: {uploaded_file.name}")
            continue
        
        try:
            loaded_docs = loader.load()
            documents.extend(loaded_docs)
            st.write(f"Loaded: {uploaded_file.name} ({len(loaded_docs)} pages/sections)")
        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {e}. Ensure 'cryptography' is installed for PDFs (pip install cryptography).")
            continue

    if not documents:
        st.error("No valid documents were loaded from your uploads.")
        return False # Indicate failure

    st.write(f"Loaded {len(documents)} document pages/sections. Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    st.write(f"Split into {len(chunks)} chunks.")

    st.write(f"Creating embeddings with {OLLAMA_EMBED_MODEL} and storing in ChromaDB...")
    try:
        embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)
        
        if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH): # Check if dir exists AND is not empty
            st.write("Loading existing ChromaDB and adding new chunks...")
            vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
            vectorstore.add_documents(chunks) # Add new chunks to the existing DB
        else:
            st.write("Creating new ChromaDB...")
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=CHROMA_PATH
            )
        
        st.success(f"ChromaDB updated with new documents at {CHROMA_PATH}")
        return True # Indicate success
    except Exception as e:
        st.error(f"Error creating/updating ChromaDB: {e}. Ensure Ollama is running and '{OLLAMA_EMBED_MODEL}' is pulled.")
        return False # Indicate failure

@st.cache_resource
def load_rag_chain_from_path(db_path):
    """
    Loads ChromaDB from a given path and builds the RAG chain.
    This function is cached and will re-run if db_path changes or cache is cleared.
    """
    if not os.path.exists(db_path) or not os.listdir(db_path):
        st.error(f"Cannot load RAG chain: ChromaDB not found at {db_path}.")
        return None, None # Return None for both vectorstore and rag_chain

    st.write(f"Loading ChromaDB from {db_path}...")
    try:
        embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)
        vectorstore_obj = Chroma(persist_directory=db_path, embedding_function=embeddings)
        
        st.write("Initializing Ollama LLM and building RAG chain...")
        llm = ChatOllama(model=OLLAMA_LLM_MODEL)
        retriever = vectorstore_obj.as_retriever(search_kwargs={"k": 3}) 

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful study assistant. Answer the user's question based *only* on the provided context. If you don't know the answer, state that you don't have enough information."),
            ("human", "Context: {context}\n\nQuestion: {input}"),
        ])

        document_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, document_chain)
        st.success("RAG chain loaded successfully!")
        return vectorstore_obj, rag_chain # Return both objects
    except Exception as e:
        st.error(f"Error loading RAG chain: {e}. Ensure Ollama is running and models are pulled.")
        return None, None # Return None for both on error

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("📚 Personalized Study Assistant")
st.markdown("Upload your study documents (PDFs, TXT, MD) to get started!")

# --- Session State Initialization ---
if "chroma_ready" not in st.session_state:
    st.session_state.chroma_ready = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "file_uploader_key" not in st.session_state:
    st.session_state.file_uploader_key = 0

# --- Initial Load of Existing ChromaDB on App Start (if not already loaded) ---
# This block runs only once if the vectorstore is not yet loaded into session state.
if st.session_state.vectorstore is None and (os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH)):
    st.info("Attempting to load previous knowledge base...")
    st.session_state.vectorstore, st.session_state.rag_chain = load_rag_chain_from_path(CHROMA_PATH)
    if st.session_state.vectorstore and st.session_state.rag_chain:
        st.session_state.chroma_ready = True
        st.success("Previous knowledge base loaded. Ready to chat!")
    else:
        st.error("Failed to load previous knowledge base. Please upload documents.")
        st.session_state.chroma_ready = False
        st.session_state.vectorstore = None
        st.session_state.rag_chain = None
else:
    # If no existing DB found initially, inform the user to upload
    if not st.session_state.chroma_ready and st.session_state.vectorstore is None:
        st.info("No existing knowledge base found. Upload documents to begin.")

# --- Sidebar for Document Management ---
with st.sidebar:
    st.header("Document Management")
    
    uploaded_files = st.file_uploader(
        "Upload your study documents", 
        type=["pdf", "txt", "md"], 
        accept_multiple_files=True,
        help="Only PDF, TXT, and MD files are supported. Uploading new files will ADD to the existing knowledge base.",
        key=f"file_uploader_{st.session_state.file_uploader_key}"
    )

    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                # This function now ONLY updates the disk DB and returns success/failure
                ingestion_success = ingest_documents_to_chromadb(uploaded_files)
                
                if ingestion_success:
                    # After successful ingestion, always try to load the RAG chain from the disk path
                    st.session_state.vectorstore, st.session_state.rag_chain = load_rag_chain_from_path(CHROMA_PATH)
                    if st.session_state.vectorstore and st.session_state.rag_chain:
                        st.session_state.chroma_ready = True
                        st.session_state.messages = [] # Clear chat history on new document upload
                        st.success("Documents processed. You can now ask questions!")
                    else:
                        st.error("Failed to load RAG chain after processing documents.")
                        st.session_state.chroma_ready = False
                else:
                    st.session_state.chroma_ready = False
                    st.session_state.rag_chain = None
                    st.session_state.vectorstore = None
            st.session_state.file_uploader_key += 1 # Increment key to reset uploader regardless of success
            st.rerun() # Ensure UI updates correctly
        else:
            st.warning("Please upload documents before clicking 'Process Documents'.")

    # # Reset button: This is the ONLY way to delete the ChromaDB
    # if st.button("Clear All Data & Restart"):
    #     reset_application_state()
    #     st.session_state.file_uploader_key += 1
    #     st.rerun()

# --- Main Chat Interface ---
if st.session_state.chroma_ready and st.session_state.rag_chain is not None:
    st.success("Knowledge base ready. Ask your questions!")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Your question..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Thinking..."):
            try:
                if st.session_state.rag_chain:
                    response = st.session_state.rag_chain.invoke({"input": prompt})
                    assistant_response = response["answer"]
                else:
                    assistant_response = "RAG chain not initialized. Please upload documents and process them first."
                    st.error(assistant_response)
            except Exception as e:
                assistant_response = f"An error occurred while getting the answer: {e}. Please ensure Ollama is running and models are loaded."
                st.error(assistant_response)

        with st.chat_message("assistant"):
            st.markdown(assistant_response)
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

        with st.expander("Show Retrieved Context"):
            if "context" in response and response["context"]:
                for i, doc in enumerate(response["context"]):
                    st.write(f"**Document {i+1}**")
                    st.write(doc.page_content)
                    if 'source' in doc.metadata:
                        st.write(f"Source: {doc.metadata['source']}")
            else:
                st.write("No context retrieved for this query.")
else:
    # This message is shown if no DB is found initially, or after a reset/failed load
    if not st.session_state.chroma_ready: # Only show this if not ready
        st.info("Please upload documents and click 'Process Documents' to start.")