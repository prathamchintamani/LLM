import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- CONFIGURATION ---
VECTOR_STORE_PATH = "vector_store"

def main():
    """
    Main function to run the RAG chat application.
    """
    print("--- RAG Chatbot for SparkleBot Inc. ---")

    # 1. CHECK FOR PREREQUISITES
    if "GOOGLE_API_KEY" not in os.environ:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        return
    if not os.path.exists(VECTOR_STORE_PATH):
        print(f"Error: Vector store not found at '{VECTOR_STORE_PATH}'.")
        print("Please run `create_index.py` first to create the index.")
        return

    # 2. LOAD THE SAVED VECTOR STORE
    print("Loading vector store...")
    # It's crucial to use the same embedding model you used to create the store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    print("Vector store loaded successfully.")

    # 3. INITIALIZE THE LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

    # 4. CREATE THE RETRIEVER
    # This object is responsible for fetching documents from the vector store.
    retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 results

    # 5. CREATE THE PROMPT TEMPLATE
    # This is the heart of RAG. We instruct the LLM on how to use the context.
    prompt_template = """
    You are an assistant for SparkleBot Inc. employees.
    Answer the question based ONLY on the following context.
    If you don't know the answer, just say that you don't know. Do not make up an answer.

    CONTEXT:
    {context}

    QUESTION:
    {input}
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # 6. CREATE THE RAG CHAIN
    # This chain will automatically:
    # 1. Take the user's question.
    # 2. Pass it to the retriever to get relevant documents.
    # 3. "Stuff" the documents and the question into the prompt.
    # 4. Pass the final prompt to the LLM to get an answer.
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # 7. START THE INTERACTIVE CHAT LOOP
    print("\nReady to answer your questions. Type 'exit' to quit.")
    while True:
        user_question = input("\nYou: ")
        if user_question.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        print("SparkleBot is thinking...")
        # Invoke the chain with the user's question
        response = retrieval_chain.invoke({"input": user_question})

        # Print the answer
        print(f"\nSparkleBot: {response['answer']}")

        # (Optional) See the source documents it used
        # print("\n--- Sources Used ---")
        # for doc in response["context"]:
        #     print(doc.page_content)
        # print("--------------------")

if __name__ == "__main__":
    main()
