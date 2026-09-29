import os
import sys
from dotenv import load_dotenv

load_dotenv()

import groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

# 1. Verify Groq API Key
groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    print("\n[ERROR] GROQ_API_KEY is not set in your .env file.\n")
    sys.exit(1)

# 2. Find a model that actively supports tool calling
client = groq.Groq(api_key=groq_key)
all_models = [m.id for m in client.models.list().data]

selected_model = None
supports_tools = False

print("\nScanning models on your Groq key...")

for model_id in all_models:
    if any(bad in model_id.lower() for bad in ["whisper", "guard", "safeguard", "embed"]):
        continue
    try:
        client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "hi"}],
            tools=[{
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "test",
                    "parameters": {"type": "object", "properties": {}},
                },
            }],
            max_tokens=1,
        )
        selected_model = model_id
        supports_tools = True
        print(f"[+] Found tool-calling compatible model: {selected_model}\n")
        break
    except Exception:
        continue

# Fallback to any active chat model if tool calling is unsupported
if not selected_model:
    for model_id in all_models:
        if any(bad in model_id.lower() for bad in ["whisper", "guard"]):
            continue
        try:
            client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
            )
            selected_model = model_id
            supports_tools = False
            print(f"[!] Tool-calling unsupported. Using standard chat model: {selected_model}\n")
            break
        except Exception:
            continue

if not selected_model:
    print("[ERROR] No working chat models available on this Groq account.")
    sys.exit(1)

# 3. Load and Split Document
pdf_path = "session6.pdf"
if not os.path.exists(pdf_path):
    print(f"\n[ERROR] Could not find {pdf_path} in current directory.")
    sys.exit(1)

loader = PyPDFLoader(pdf_path)
docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(loader.load())

# 4. Embeddings & Vector Store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = InMemoryVectorStore.from_documents(documents=docs, embedding=embeddings)

# 5. Initialize LLM
llm = ChatGroq(
    model=selected_model,
    temperature=0,
    api_key=groq_key
)

# 6. Search Tool & Agent Setup
@tool
def retrieve_context(query: str) -> str:
    """Retrieve documents relevant to a query from the knowledge base."""
    results = vector_db.similarity_search(query=query, k=3)
    return "\n\n".join(doc.page_content for doc in results)

system_prompt = (
    "You are a helpful assistant that answers questions using retrieved context from session6.pdf. "
    "Always rely on the retrieved context to answer accurately."
    "Output plain text only. Do not use LaTeX, Markdown, or any other formatting or even math blocks or backlashes for formulas."
)

if supports_tools:
    agent = create_react_agent(
        model=llm,
        tools=[retrieve_context],
        prompt=system_prompt,
        checkpointer=InMemorySaver()
    )

# 7. Interactive Terminal Chat Loop
print("=" * 50)
print("  RAG Agent Ready. Type your question below (or 'exit' to quit)")
print("=" * 50)

while True:
    try:
        query = input("\nUser: ")
        if query.lower().strip() == "exit":
            break
        if not query.strip():
            continue

        if supports_tools:
            response = agent.invoke(
                {"messages": [{"role": "user", "content": query}]},
                config={"configurable": {"thread_id": "session_1"}}
            )
            result = response["messages"][-1].content
        else:
            search_results = vector_db.similarity_search(query=query, k=3)
            context = "\n\n".join(doc.page_content for doc in search_results)
            full_prompt = (
                f"{system_prompt}\n\n"
                f"Retrieved Context:\n{context}\n\n"
                f"User Question: {query}\n\n"
                f"Answer:"
            )
            result = llm.invoke(full_prompt).content

        print("\nAssistant:")
        print("-" * 50)
        print(result)
        print("=" * 50)

    except KeyboardInterrupt:
        print("\nExiting session...")
        break