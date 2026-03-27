import json
import faiss
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer

MODEL_NAME = "qwen2.5:7b"
EMBED_MODEL = "/home/ubuntu/.cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots/e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
VECTOR_STORE_PATH = "vector_store"
MIN_SCORE = 0.35
TOP_K = 3
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
        f"[Источник: {c['metadata']['filename']} | chunk_id: {c['metadata']['chunk_id']}]\n{c['text']}"
        for c in retrieved_chunks
    ])

    return f"""Ты помощник по базе знаний Quantum Forge.

Правила ответа:
- Отвечай только на русском языке.
- Используй только информацию из контекста.
- Не добавляй внешние знания.
- Если точного ответа нет, напиши: Я не знаю.
- Ответ должен быть коротким.
- Формат строго такой:

Шаги:
1. ...
2. ...
3. ...

Ответ: ...

Ниже примеры правильного поведения.

Пример 1:
Вопрос: Кто такой Арин Вейл?
Шаги:
1. Нахожу описание Арина Вейла в контексте.
2. В тексте указано, что он оператор биосинтов из Нова-Прайм.
3. Формирую краткий ответ по найденному фрагменту.
Ответ: Арин Вейл — оператор биосинтов из Нова-Прайм, расположенного в Секторе К-1. Его цель — стать мастером управления биосинтами.

Пример 2:
Вопрос: Что такое Орден Нуль?
Шаги:
1. Ищу в контексте описание Ордена Нуль.
2. В тексте указано, что это нелегальная сеть.
3. Формирую краткий ответ без внешних добавлений.
Ответ: Орден Нуль — нелегальная сеть, действующая в различных секторах с целью контроля над биосинтами и энергетическими ресурсами.

Контекст:
{context}

Вопрос: {query}
"""


def ask_llm(prompt):
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "Отвечай только на русском языке. Не продолжай ответ после строки, начинающейся с 'Ответ:'."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0.1,
            "top_p": 0.8,
            "num_predict": 120,
            "stop": ["\n\n\n", "Контекст:", "Вопрос:"]
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
            print(r["text"][:200])

    if not retrieved:
        return "Я не знаю."

    max_score = max(r["score"] for r in retrieved)

    if max_score < MIN_SCORE:
        return "Я не знаю."

    prompt = build_prompt(query, retrieved)
    answer = ask_llm(prompt)
    return answer


def main():
    print("RAG-бот Quantum Forge запущен.")
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