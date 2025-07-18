import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

KNOWLEDGE_BASE_FILE = "InterIIT_rulebook_assistant/rulebook.pdf"
VECTOR_STORE_PATH = "InterIIT_rulebook_assistant/vector_store"

def main():
    """
    This script creates a searchable vector store from a text document.
    """
    print("--- Starting Index Creation ---")

    if "GOOGLE_API_KEY" not in os.environ:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        return

    print(f"Loading document: {KNOWLEDGE_BASE_FILE}")
    loader = PyPDFLoader(KNOWLEDGE_BASE_FILE)
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s).")

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    docs = text_splitter.split_documents(documents)
    print(f"Split into {len(docs)} chunks.")

    print("Creating embeddings and building the vector store...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    vector_store = FAISS.from_documents(docs, embeddings)
    print("Vector store created in memory.")

    print(f"Saving vector store to disk at: {VECTOR_STORE_PATH}")
    vector_store.save_local(VECTOR_STORE_PATH)
    print("--- Index Creation Complete ---")

if __name__ == "__main__":
    main()
