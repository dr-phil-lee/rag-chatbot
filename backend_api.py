"""
RAG backend API for handbook chatbot.

Run:
  pip install -U flask flask-cors
  python backend_api.py
"""

import os
import re
import tempfile
from typing import List, Optional, Tuple

import pdfplumber
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_full_text(pdf_path: str) -> List[dict]:
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            text = normalize_text(page.extract_text())
            if text:
                texts.append({"page": page_idx, "text": text})
    return texts


def build_vectorstore(pages: List[dict], embedding_model) -> Tuple[FAISS, int]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    documents = []
    chunk_id = 0
    for page in pages:
        for split in splitter.split_text(page["text"]):
            documents.append(
                Document(
                    page_content=split,
                    metadata={
                        "page": page["page"],
                        "chunk_id": chunk_id,
                        "citation": f"Page {page['page'] + 1}",
                    },
                )
            )
            chunk_id += 1
    return FAISS.from_documents(documents, embedding_model), len(documents)


def run_rag(question: str, retriever, reranker, llm, top_k: int = 3):
    retrieved_docs = retriever.invoke(question)
    pairs = [[question, doc.page_content] for doc in retrieved_docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, _ in ranked[:top_k]]

    context = "\n\n".join(
        f"[Ref {i}] Page {doc.metadata['page'] + 1}\n{doc.page_content.strip()}"
        for i, doc in enumerate(top_docs, start=1)
    )

    prompt = ChatPromptTemplate.from_template(
        """
Use ONLY the provided context.
Do NOT make up facts.
Answer briefly in 1-2 sentences.
Quote the exact policy sentence when possible.
Include citations like [Ref 1].
If the context does not contain the answer, say: "I cannot provide the answer."

Context:
{context}

Question:
{question}

ANSWER:
"""
    )

    chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    raw = chain.invoke({"context": context, "question": question})
    cleaned = raw.split("Assistant:")[0].strip()
    match = re.search(r"ANSWER:\s*(.*)", cleaned, re.DOTALL)
    answer = match.group(1).strip() if match else cleaned
    answer = answer.split("REFERENCES:")[0].strip()
    answer = re.sub(r"\s+", " ", answer)

    refs = [
        {
            "idx": i,
            "page": doc.metadata["page"] + 1,
            "snippet": doc.page_content[:280].replace("\n", " "),
        }
        for i, doc in enumerate(top_docs, start=1)
    ]
    return answer, refs


class RAGService:
    def __init__(self):
        self.embedding_model = None
        self.reranker = None
        self.llm = None
        self.retriever = None
        self.index_info = {"n_pages": 0, "n_chunks": 0, "filename": ""}

    def ensure_models_loaded(self):
        if self.embedding_model is None:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        if self.reranker is None:
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        if self.llm is None:
            model_name = "Qwen/Qwen2.5-7B-Instruct"
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
            )
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                do_sample=False,
                max_new_tokens=180,
                return_full_text=False,
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)

    def build_index(self, pdf_bytes: bytes, filename: str):
        self.ensure_models_loaded()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(pdf_bytes)
            temp_path = temp_pdf.name
        try:
            pages = extract_full_text(temp_path)
            vectorstore, n_chunks = build_vectorstore(pages, self.embedding_model)
            self.retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
            self.index_info = {
                "n_pages": len(pages),
                "n_chunks": n_chunks,
                "filename": filename,
            }
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def ask(self, question: str, top_k: int):
        if self.retriever is None:
            raise ValueError("PDF index not built yet. Upload a PDF first.")
        if self.reranker is None or self.llm is None:
            self.ensure_models_loaded()
        return run_rag(question, self.retriever, self.reranker, self.llm, top_k=top_k)

    def clear(self):
        self.retriever = None
        self.index_info = {"n_pages": 0, "n_chunks": 0, "filename": ""}


app = Flask(__name__)
CORS(app)
rag = RAGService()


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/upload")
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"detail": "No file provided."}), 400
    file = request.files["file"]
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"detail": "Only PDF files are allowed."}), 400
    pdf_bytes = file.read()
    if not pdf_bytes:
        return jsonify({"detail": "Uploaded PDF is empty."}), 400

    try:
        rag.build_index(pdf_bytes, file.filename)
    except Exception as exc:
        return jsonify({"detail": f"Indexing failed: {str(exc)}"}), 500

    return jsonify(
        {
            "message": "Index built successfully.",
            "n_pages": rag.index_info["n_pages"],
            "n_chunks": rag.index_info["n_chunks"],
            "filename": rag.index_info["filename"],
        }
    )


@app.post("/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    question: Optional[str] = payload.get("question")
    top_k: int = int(payload.get("top_k", 3))

    if not question or not question.strip():
        return jsonify({"detail": "question is required."}), 400

    top_k = max(1, min(top_k, 5))
    try:
        answer, refs = rag.ask(question.strip(), top_k=top_k)
    except ValueError as exc:
        return jsonify({"detail": str(exc)}), 400
    except Exception as exc:
        return jsonify({"detail": f"Chat failed: {str(exc)}"}), 500

    return jsonify({"answer": answer, "refs": refs})


@app.post("/clear")
def clear():
    rag.clear()
    return jsonify({"message": "Chat/index state cleared."})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
