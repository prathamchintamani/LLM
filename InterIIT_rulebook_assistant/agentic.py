import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain import hub

VECTOR_STORE_PATH = "InterIIT_rulebook_assistant/vector_store"

def main():
    print("agentic assistant for interiit rulebook")
    if "GOOGLE_API_KEY" not in os.environ:
        print("API key not loaded correctly")
        return
    if not os.path.exists(VECTOR_STORE_PATH):
        print(f"Error: Vector store not found at '{VECTOR_STORE_PATH}'.")
        print("Please run `create_index.py` first to create the index.")
        return

    print("loading vector store")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    print("loaded vector store")

    retriever = vector_store.as_retriever()
    llm = ChatGoogleGenerativeAI(model = "gemini-2.0-flash")
    prompt_template = """
    You are an assistant for participants of the 13th Inter IIT techmeet,
    Answer the question based ONLY on the following context.
    If you don't know the answer, just say that you don't know. Do not make up an answer.

    CONTEXT:
    {context}

    QUESTION:
    {input}
    """
    rag_prompt = ChatPromptTemplate.from_template(prompt_template)
    document_chain = create_stuff_documents_chain(llm,rag_prompt)
    rag_chain = create_retrieval_chain(retriever,document_chain)

    # --- ADD THIS NEW SECTION AFTER CREATING rag_chain ---
    tools = [
        Tool(
            name="Inter IIT rulebook assistant",
            func=lambda query: rag_chain.invoke({"input": query}),
            description="use this for questions regarding rules and proceedings of InterIIT Tech Meet 13.0 going to be held in IIT bombay."
        ),
        Tool(
            name="General Knowledge Search",
            func=llm.invoke,
            description="Use this for all other general questions not related to SparkleBot Inc."
        ),
    ]

    agent_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, agent_prompt)
    agent_executor = AgentExecutor(agent = agent, tools = tools, verbose= True)

    print("\nReady... Type 'exit' to quit.")
    while True:
        user_question = input("\nYou: ")
        if user_question.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        print("Retrieving and Generating ...")
        # Invoke the chain with the user's question
        response = agent_executor.invoke({"input": user_question})

        # Print the answer
        print(f"\nRulebook: {response['output']}")

if __name__ == "__main__":
    main()
