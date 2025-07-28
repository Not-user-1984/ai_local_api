from fastapi import FastAPI, UploadFile, File, Form
from llama_cpp import Llama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
import fitz
import os

app = FastAPI()

# === Загружаем LLM ===
llm_general = Llama(model_path="models/mistral.gguf", n_ctx=512, n_threads=6)
llm_code = Llama(model_path="models/codellama.gguf", n_ctx=512, n_threads=6)

# === Инициализация Chroma ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="embeddings", embedding_function=embedding_model)

# === Обработка текста PDF ===
def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)

def ingest_pdf_to_chroma(file_path: str):
    text = extract_text_from_pdf(file_path)
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)
    vector_db.add_texts(chunks)
    vector_db.persist()

# === API ===

@app.post("/chat/general")
def general_chat(prompt: str):
    out = llm_general(prompt, max_tokens=200)
    return {"response": out["choices"][0]["text"].strip()}

@app.post("/chat/code")
def code_chat(prompt: str):
    out = llm_code(prompt, max_tokens=200)
    return {"response": out["choices"][0]["text"].strip()}

@app.post("/rag/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    path = f"docs/{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())
    ingest_pdf_to_chroma(path)
    return {"status": "PDF загружен и обработан", "filename": file.filename}

@app.post("/rag/ask")
def ask_rag(prompt: str = Form(...)):
    results = vector_db.similarity_search(prompt, k=3)
    context = "\n".join([doc.page_content for doc in results])
    full_prompt = f"Ответь на основе текста:\n{context}\n\nВопрос: {prompt}\nОтвет:"
    out = llm_general(full_prompt, max_tokens=200)
    return {"response": out["choices"][0]["text"].strip()}
