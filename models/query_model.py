import numpy as np
from functools import lru_cache

from models import load_models

class AskModel:

    SYSTEM_PROMPT = (
        "You are a helpful and reliable Japan travel assistant. "
        "Answer concisely and only using the information provided in the passages. "
        "Answer in complete sentences. "
        "Summarize key recommendations clearly in paragraph form. Do not reproduce lists or bullet points."
        "Do not speculate or include unrelated content. "
        "If the answer is not in the passages, say: 'I couldn’t find that information.'"
    )

    def __init__(self, collection):
        mods = load_models.LoadModels()
        self.collection = collection
        self.embedder = mods.embedder
        self.cross_encoder = mods.cross_encoder
        self.tokenizer = mods.tokenizer
        self.model = mods.load_model()

    @staticmethod
    def truncate(text, max_words=100):
        return " ".join(text.split()[:max_words])

    @staticmethod
    def trim_to_token_limit(texts, tokenizer, max_tokens=1500):
        tokenized_chunks = [tokenizer(t, add_special_tokens=False)["input_ids"] for t in texts]

        total_tokens = 0
        included = []
        for chunk, tokens in zip(texts, tokenized_chunks):
            if total_tokens + len(tokens) > max_tokens:
                break
            included.append(chunk)
            total_tokens += len(tokens)

        return "\n\n---\n\n".join(included)

    @lru_cache(maxsize=1024)
    def get_query_embedding(self, query: str) -> np.ndarray:
        emb = self.embedder.encode([query], convert_to_numpy=True)
        return emb / np.linalg.norm(emb, axis=1, keepdims=True)

    @lru_cache(maxsize=512)
    def answer_cached(self, query: str, temperature: float, top_p: float, max_new_tokens: int,
                      repitition_penalty: float) -> str:
        q_emb = self.get_query_embedding(query).tolist()
        results = self.collection.query(query_embeddings=q_emb, n_results=10,
                                   include=["documents", "metadatas"])

        if not results["documents"] or not results["documents"][0]:
            return "I couldn’t find that information."

        docs = [{"text": t, "url": m["url"]} for t, m in zip(results["documents"][0], results["metadatas"][0])]
        pairs = [(query, d["text"]) for d in docs]
        scores = self.cross_encoder.predict(pairs)
        top_docs = [d for d, _ in sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:5]]

        docs_text = self.trim_to_token_limit([self.truncate(d["text"]) for d in top_docs], self.tokenizer)

        prompt = (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"Passages:\n{docs_text}\n\n"
            f"Question: {query}\n\n"
            f"Answer in one paragraph: Answer:"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repitition_penalty,
            eos_token_id=self.tokenizer.eos_token_id
        )

        output_text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        if "Answer:" in output_text:
            return output_text.split("Answer:")[-1].strip()
        return output_text.strip()