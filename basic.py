import os
from langchain_google_genai import ChatGoogleGenerativeAI

def main():
    """
    Main function to run the simple LLM chat application.
    """
    print("--- Welcome to the Gemini LLM Terminal ---")
    print("The script will use your GOOGLE_API_KEY from your environment variables.")
    print("Type 'exit' or 'quit' to end the session.\n")

    # 1. Initialize the LLM
    # We specify the model name. `gemini-1.5-flash-latest` points to the most recent version.
    # The library will automatically find and use your GOOGLE_API_KEY.
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
    except Exception as e:
        print(f"Error initializing the LLM. Is your GOOGLE_API_KEY set correctly? Error: {e}")
        return

    # 2. Create an interactive loop
    while True:
        question = input("you: ")

        if question.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        # 3. Invoke the model with your question
        # This is the line that actually makes the API call to Google.
        print("\nGemini is thinking...")
        response = llm.invoke(question)

        # 4. Print the response content
        # The 'response' object is a special LangChain message object.
        # We access the actual text content using the .content attribute.
        print(f"\nGemini: {response.content}\n")

if __name__ == "__main__":
    main()
