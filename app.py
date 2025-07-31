from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import os
import threading
import requests
from dotenv import load_dotenv
from datetime import datetime
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# 🎙️ Friendly tone prompt
system_template = (
    "You are Genie, a friendly product expert for a B2B mobile accessories platform. "
    "You speak casually like a helpful friend of the retailer. Always keep replies short, clear, and product-focused. "
    "Ask follow-up questions if you need more info to assist. "
    "If you're unsure about the answer or it’s outside product details, say:\n"
    "'I'm not sure about this. You can call or WhatsApp our support team for help.'"
)

system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_context_prompt = HumanMessagePromptTemplate.from_template("Relevant info:\n{context}")
human_question_prompt = HumanMessagePromptTemplate.from_template("{question}")

chat_prompt = ChatPromptTemplate.from_messages([
    system_message_prompt,
    human_context_prompt,
    human_question_prompt
])

# Load env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load and embed documents
def load_all_texts(data_dir):
    all_docs = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".txt"):
                path = os.path.join(root, file)
                loader = TextLoader(path, encoding="utf-8")
                docs = loader.load()
                all_docs.extend(docs)
    return all_docs

print("📄 Loading documents...")
raw_docs = load_all_texts("Chatbot_data")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(raw_docs)
print(f"🧠 Total chunks: {len(docs)}")

print("🔗 Creating FAISS index...")
vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings(openai_api_key=openai_api_key))
retriever = vectorstore.as_retriever()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, openai_api_key=openai_api_key)

# 🔁 QA Chain setup
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    combine_docs_chain_kwargs={"prompt": chat_prompt},
    verbose=True
)

# === API setup ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    chat_history: list

@app.post("/chat")
def chat(query: QueryRequest):
    # 🧠 Format chat history
    history = query.chat_history
    formatted_history = []
    for i in range(0, len(history) - 1, 2):
        user_msg = history[i]["content"]
        ai_msg = history[i + 1]["content"]
        formatted_history.append((user_msg, ai_msg))

    # 🎯 Run the QA chain
    result = qa_chain.invoke({
        "question": query.question,
        "chat_history": formatted_history
    })

    answer = result["answer"]

    # 🧾 Log question & answer with timestamp
    threading.Thread(
        target=log_to_google_sheets,
        args=(query.question, answer, "anonymous"),
        daemon=True
    ).start()

    return {"response": answer}

# === Logging function ===
def log_to_google_sheets(question: str, answer: str, user: str):
    try:
        timestamp = datetime.now().isoformat()
        user = "Anonymous"
        
        payload = {
                "timestamp": timestamp,
                "user": user,
                "question": question,
                "answer": answer
        }

        response = requests.post(
            "https://script.google.com/macros/s/AKfycbyWYAokv_kJJjTcpxEMxGxUKHJqoJQAVwT4tdmfV47kwFRQO6gNNptJSAsIPlHTjQi1/exec",
            json={
        )
        
        if response.status_code != 200:
            print(f"⚠️ Google Sheets logging failed: {response.status_code}")
    except Exception as e:
        print("❌ Logging error:", e)
