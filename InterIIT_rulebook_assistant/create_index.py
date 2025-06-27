import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# --- CONFIGURATION ---
KNOWLEDGE_BASE_FILE = "InterIIT_rulebook_assistant/rulebook.pdf"
VECTOR_STORE_PATH = "InterIIT_rulebook_assistant/vector_store"

def main():
    """
    This script creates a searchable vector store from a text document.
    """
    print("--- Starting Index Creation ---")

    # Check for API key
    if "GOOGLE_API_KEY" not in os.environ:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        return

    # 1. LOAD THE DOCUMENT
    print(f"Loading document: {KNOWLEDGE_BASE_FILE}")
    loader = PyPDFLoader(KNOWLEDGE_BASE_FILE)
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s).")

    # 2. SPLIT THE DOCUMENT INTO CHUNKS
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # The max number of characters in a chunk
        chunk_overlap=100   # The number of characters to overlap between chunks
    )
    docs = text_splitter.split_documents(documents)
    print(f"Split into {len(docs)} chunks.")

    # 3. EMBED AND STORE THE CHUNKS
    print("Creating embeddings and building the vector store...")
    # This uses the "models/embedding-001" model by default
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # FAISS.from_documents does two things:
    # 1. It calls the embedding model for each chunk.
    # 2. It stores the chunk and its embedding in the FAISS vector store.
    vector_store = FAISS.from_documents(docs, embeddings)
    print("Vector store created in memory.")

    # 4. SAVE THE VECTOR STORE TO DISK
    print(f"Saving vector store to disk at: {VECTOR_STORE_PATH}")
    vector_store.save_local(VECTOR_STORE_PATH)
    print("--- Index Creation Complete ---")

if __name__ == "__main__":
    main()
