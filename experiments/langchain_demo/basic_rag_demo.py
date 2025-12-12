import os
import sys

# Minimal check for dependencies
try:
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
except ImportError as e:
    print(f"Error: Missing dependencies. Please run 'pip install -r experiments/langchain_demo/requirements.txt'.\nDetails: {e}")
    sys.exit(1)

def run_demo():
    """
    Demonstrates a basic RAG pipeline using LangChain Expression Language (LCEL).
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\n[!] OPENAI_API_KEY environment variable not found.")
        print("    The script cannot execute the actual LLM call, but the code structure below")
        print("    demonstrates the correct implementation of a modern LangChain RAG pipeline.")
        return

    print("Initializing RAG Pipeline...")

    # 1. Load Data (Using a dummy text for demo purposes)
    # In a real scenario, this would load from the project's data sources.
    dummy_text = """
    BeyondChats is an AI-first chatbot company. 
    Our mission is to help buyers make better decisions and help brands generate more qualified leads.
    We are looking for a talented LLM Engineer intern who is obsessed with experimenting with AI technologies.
    """
    
    # Save dummy text to a temporary file for the loader
    with open("temp_demo_data.txt", "w") as f:
        f.write(dummy_text)

    loader = TextLoader("temp_demo_data.txt")
    docs = loader.load()

    # 2. Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    splits = text_splitter.split_documents(docs)

    # 3. Embed & Store
    # Note: This requires an API key for embeddings unless using a local alternative like HuggingFaceEmbeddings
    vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    # 4. Define Prompt
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 5. Initialize Model
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # 6. Build Chain (LCEL)
    # This is the modern "Pro" way to build chains in LangChain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 7. Execute
    question = "What is the mission of BeyondChats?"
    print(f"\nQuestion: {question}")
    response = rag_chain.invoke(question)
    print(f"Answer: {response}")

    # Cleanup
    if os.path.exists("temp_demo_data.txt"):
        os.remove("temp_demo_data.txt")

if __name__ == "__main__":
    run_demo()
