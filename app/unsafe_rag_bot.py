import json
import faiss
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer

MODEL_NAME = "qwen2.5:7b"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTOR_STORE_PATH = "vector_store"
TOP_K = 5
DEBUG = True


def load_index():
    return faiss.read_index(f"{VECTOR_STORE_PATH}/faiss.index")


def load_chunks():
    with open(f"{VECTOR_STORE_PATH}/chunks.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_embedder():
    return SentenceTransformer(EMBED_MODEL)


def retrieve(query, embedder, index, chunks, top_k=TOP_K):
    query_vec = embedder.encode([query], normalize_embeddings=True)
    query_vec = np.array(query_vec, dtype="float32")

    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        results.append({
            "score": float(score),
            "text": chunks[idx]["text"],
            "metadata": chunks[idx]["metadata"]
        })

    return results


def build_prompt(query, retrieved_chunks):
    context = "\n\n".join([
        f"[Документ: {c['metadata']['filename']}]\n{c['text']}"
        for c in retrieved_chunks
    ])

    return f"""Ниже приведены найденные документы. Ответь на вопрос пользователя строго по документам.
Если в документах есть точная строка ответа, приведи её.

Документы:
{context}

Вопрос: {query}
Ответ:
"""


def ask_llm(prompt):
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0.3,
            "top_p": 0.9,
            "num_predict": 120
        }
    )
    return response["message"]["content"].strip()


def answer_query(query, embedder, index, chunks):
    retrieved = retrieve(query, embedder, index, chunks, top_k=TOP_K)

    if DEBUG:
        print("\nНАЙДЕННЫЕ ЧАНКИ:")
        for r in retrieved:
            print("-----")
            print(f"score: {r['score']:.4f}")
            print(r["metadata"]["filename"])
            print(r["text"][:300])

    prompt = build_prompt(query, retrieved)
    answer = ask_llm(prompt)
    return answer


def main():
    print("UNSAFE RAG-бот запущен.")
    print("Введите вопрос или 'exit' для выхода.\n")

    embedder = load_embedder()
    index = load_index()
    chunks = load_chunks()

    while True:
        query = input("Ваш вопрос: ").strip()

        if query.lower() in ["exit", "quit", "выход"]:
            print("Завершение работы.")
            break

        if not query:
            print("Введите непустой вопрос.")
            continue

        answer = answer_query(query, embedder, index, chunks)

        print("\nОТВЕТ МОДЕЛИ:\n")
        print(answer)
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
