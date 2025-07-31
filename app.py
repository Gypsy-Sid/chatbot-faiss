import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory

# === Load env ===
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# === Constants ===
DATA_DIR = "chatbot_data"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# === Load and Embed ===
def load_all_texts(data_dir):
    all_docs = []
    print(f"🔍 Scanning folder: {data_dir}")
    for root, _, files in os.walk(data_dir):
        print(f"📁 Checking: {root}")
        for file in files:
            print(f"📄 Found file: {file}")
            if file.endswith(".txt"):
                path = os.path.join(root, file)
                print(f"✅ Loading: {path}")
                loader = TextLoader(path, encoding="utf-8")
                docs = loader.load()
                all_docs.extend(docs)
    return all_docs


print("📄 Loading documents...")
raw_docs = load_all_texts(DATA_DIR)
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
docs = splitter.split_documents(raw_docs)
print(f"🧠 Total chunks: {len(docs)}")

print("🔗 Creating FAISS index...")
vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings(openai_api_key=openai_api_key))
retriever = vectorstore.as_retriever()

# === LLM setup ===
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, openai_api_key=openai_api_key)
memory = ConversationBufferWindowMemory(k=3, return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    verbose=True
)

# === FastAPI setup ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    chat_history: list = []

@app.post("/chat")
def chat(query: QueryRequest):
    result = qa_chain.invoke({"question": query.question, "chat_history": query.chat_history})
    return {"response": result["answer"]}
