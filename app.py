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

# 🎙️ Improved prompt for memory and crispness
system_template = (
    "You are Genie, a helpful product expert for a B2B mobile accessories platform. "
    "Speak casually like a friendly assistant. Always answer in short, crisp sentences — unless a product description requires more detail. "
    "Ask follow-up questions if you need more info to assist. "
    "If the user asks vague follow-ups like 'kitne ka hai' or 'ye acha hai kya', you must refer back to the last product discussed in the conversation."
    "If you're unsure or the question is unrelated to products, respond with:\n"
    "'I'm not sure about this. You can call or WhatsApp our support team on +91-9119077752 for help.'\n"
    "If you recommend products, ONLY respond in this strict format below after you have given the text explaination. No markdown. No extra text. No explanation:\n\n"
    "**Product Name**\n**https://product-link.com**\n\nRepeat for each item. Do not include any other sentence."
    "One product per pair of lines. No markdown links. No extras."
)

system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_context_prompt = HumanMessagePromptTemplate.from_template("Refer to this context if needed:\n{context}")
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
    history = query.chat_history
    formatted_history = []
    for i in range(0, len(history) - 1, 2):
        user_msg = history[i]["content"]
        ai_msg = history[i + 1]["content"]
        formatted_history.append((user_msg, ai_msg))

    # Pass only recent turns
    formatted_history = formatted_history[-6:]

    # Inject last known product context if found
    last_product_context = ""
    for turn in reversed(formatted_history):
        if "₹" in turn[1] or "View Product:" in turn[1] or "backup" in turn[1].lower():
            last_product_context = turn[1]
            break

    result = qa_chain.invoke({
        "question": query.question,
        "chat_history": formatted_history,
        "context": last_product_context
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
            json=payload
        )

        if response.status_code != 200:
            print(f"⚠️ Google Sheets logging failed: {response.status_code}")
    except Exception as e:
        print("❌ Logging error:", e)
