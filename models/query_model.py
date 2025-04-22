import numpy as np
from functools import lru_cache

from models import load_models

class AskModel:
    """
    Combines retrieval, reranking, and generation to answer user questions.

    Attributes:
        SYSTEM_PROMPT (str): System instruction for LLM behavior.
        collection (ChromaDB collection): Embedded knowledge base.
    """

    SYSTEM_PROMPT = (
        "You are a helpful and reliable Japan travel assistant. "
        "Answer concisely and only using the information provided in the passages. "
        "Answer in complete sentences. "
        "Summarize key recommendations clearly in paragraph form. "
        "Do not speculate or include unrelated content. "
        "If the answer is not in the passages, say: 'I couldn’t find that information.'"
    )

    def __init__(self, collection):
        """
        Initialize model pipeline for answering questions.

        Args:
            collection (ChromaDB collection): Source of retrieved passages.
        """
        mods = load_models.LoadModels()
        self.collection = collection
        self.embedder = mods.embedder
        self.cross_encoder = mods.cross_encoder
        self.tokenizer = mods.tokenizer
        self.model = mods.load_model()

    @staticmethod
    def truncate(text, max_words=100):
        """
        Trim a passage to a maximum number of words.

        Args:
            text (str): Input passage.
            max_words (int): Word limit.

        Returns:
            str: Truncated passage.
        """
        return " ".join(text.split()[:max_words])

    @staticmethod
    def trim_to_token_limit(texts, tokenizer, max_tokens=1500):
        """
        Join multiple texts without exceeding the token limit.

        Args:
            texts (list[str]): List of texts to join.
            tokenizer (Tokenizer): HuggingFace tokenizer.
            max_tokens (int): Max token count allowed.

        Returns:
            str: Joined texts within token constraints.
        """
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
        """
        Encode a query string into a normalized embedding.

        Args:
            query (str): User query.

        Returns:
            np.ndarray: Normalized query embedding.
        """
        emb = self.embedder.encode([query], convert_to_numpy=True)
        return emb / np.linalg.norm(emb, axis=1, keepdims=True)

    @lru_cache(maxsize=512)
    def answer_cached(self, query: str, temperature: float, top_p: float, max_new_tokens: int,
                      repitition_penalty: float) -> str:
        """
        Main RAG pipeline: retrieve, rerank, and generate an answer.

        Args:
            query (str): User input.
            temperature (float): Generation temperature.
            top_p (float): Nucleus sampling.
            max_new_tokens (int): Max new tokens to generate.
            repitition_penalty (float): Penalty for repetition.

        Returns:
            str: Final model-generated answer.
        """
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