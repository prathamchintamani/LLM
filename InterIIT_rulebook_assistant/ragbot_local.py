import os
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

VECTOR_STORE_PATH = "InterIIT_rulebook_assistant/vector_store_local"

def main():

    print("--- Assistant for InterIIT 14.0 Rulebook ---")

    if not os.path.exists(VECTOR_STORE_PATH):
        print(f"Error: Vector store not found at '{VECTOR_STORE_PATH}'.")
        print("Esteemed user, you need to first create the index by running create_index.py")
        return

    print("Loading vector store...")

    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url = "http://localhost:11434")
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    print("Vector store loaded successfully.")

    llm = ChatOllama(model="tinyllama", base_url="http://localhost:11434")

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    prompt_template = """
    You are an assistant for participants of the 13th Inter IIT techmeet,
    Answer the question based ONLY on the following context.
    Your name is djikhxtbn.
    If you don't know the answer, just say that you don't know. Do not make up an answer.

    CONTEXT:
    {context}

    QUESTION:
    {input}
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    print("\nReady... Type 'exit' to quit.")
    while True:
        user_question = input("\nYou: ")
        if user_question.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        print("Retrieving and Generating ...")

        response = retrieval_chain.invoke({"input": user_question})

        print(f"\nRulebook: {response['answer']}")

if __name__ == "__main__":
    main()
