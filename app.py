# app.py
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- Configuration ---
CHROMA_PATH = "chroma_db"
OLLAMA_LLM_MODEL = "gemma3:12b" # Your chosen LLM
OLLAMA_EMBED_MODEL = "nomic-embed-text:v1.5"

# --- Load Components ---
def load_rag_components():
    print("Loading ChromaDB...")
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 relevant chunks

    print(f"Initializing Ollama LLM: {OLLAMA_LLM_MODEL}...")
    llm = Ollama(model=OLLAMA_LLM_MODEL)

    # --- Define Prompt ---
    # This prompt tells the LLM how to answer the question based on retrieved context
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful study assistant. Answer the user's question based *only* on the provided context. If you don't know the answer, state that you don't have enough information."),
        ("human", "Context: {context}\n\nQuestion: {input}"),
    ])

    # --- Build RAG Chain ---
    # 1. Combines retrieved documents into a single string for the LLM
    document_chain = create_stuff_documents_chain(llm, prompt)

    # 2. Creates the full retrieval chain (retrieves docs, then passes to document_chain)
    rag_chain = create_retrieval_chain(retriever, document_chain)

    print("RAG components loaded successfully.")
    return rag_chain

# --- Main Application Logic ---
if __name__ == "__main__":
    rag_chain = load_rag_components()

    print("\n--- Personalized Study Assistant ---")
    print("Ask me anything about your study materials (type 'exit' to quit).")

    while True:
        query = input("\nYour question: ")
        if query.lower() == 'exit':
            break

        print("Thinking...")
        # Invoke the RAG chain
        response = rag_chain.invoke({"input": query})

        print("\nAssistant:")
        # The response object will contain 'answer' and 'context' (retrieved documents)
        print(response["answer"])

        # Optional: print retrieved context to see what the LLM used
        # print("\n--- Retrieved Context ---")
        # for i, doc in enumerate(response["context"]):
        #     print(f"Doc {i+1}: {doc.page_content[:200]}...") # Print first 200 chars
        #     print(f"Source: {doc.metadata.get('source', 'N/A')}")