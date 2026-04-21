import os
import json
import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

KNOWLEDGE_BASE_PATH = "knowledge_base"
VECTOR_STORE_PATH = "vector_store"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_documents():
    documents = []

    for filename in os.listdir(KNOWLEDGE_BASE_PATH):
        if filename.endswith(".txt"):
            filepath = os.path.join(KNOWLEDGE_BASE_PATH, filename)

            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read().strip()

            documents.append({
                "text": text,
                "metadata": {
                    "source_path": filepath,
                    "filename": filename,
                    "title": filename.replace(".txt", "")
                }
            })

    return documents


def split_documents_into_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []

    for doc in documents:
        split_texts = splitter.split_text(doc["text"])

        for i, chunk_text in enumerate(split_texts):
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source_path": doc["metadata"]["source_path"],
                    "filename": doc["metadata"]["filename"],
                    "title": doc["metadata"]["title"],
                    "chunk_id": f'{doc["metadata"]["title"]}_{i}'
                }
            })

    return chunks


def generate_embeddings(chunks):
    model = SentenceTransformer(MODEL_NAME)
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return np.array(embeddings, dtype="float32")


def save_faiss_index(embeddings):
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    index_path = os.path.join(VECTOR_STORE_PATH, "faiss.index")
    faiss.write_index(index, index_path)

    return index_path


def save_chunks(chunks):
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

    chunks_path = os.path.join(VECTOR_STORE_PATH, "chunks.json")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    return chunks_path


if __name__ == "__main__":
    documents = load_documents()
    chunks = split_documents_into_chunks(documents)
    embeddings = generate_embeddings(chunks)

    index_path = save_faiss_index(embeddings)
    chunks_path = save_chunks(chunks)

    print(f"Загружено документов: {len(documents)}")
    print(f"Создано чанков: {len(chunks)}")
    print(f"Размерность эмбеддинга: {embeddings.shape[1]}")
    print(f"FAISS индекс сохранён: {index_path}")
    print(f"Чанки сохранены: {chunks_path}")