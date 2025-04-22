import os
import torch
import asyncio

from docs import urls
from docs.load_docs import FetchDocuments
from docs.utils import ingest_into_chromadb, docs_to_passages
from models.query_model import AskModel

# Set threading limits for performance
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
torch.set_num_threads(8)

async def main():
    """
    Entry point for the Japan Travel RAG application.

    - Initializes ChromaDB and Phi-2 model.
    - Crawls and ingests documents if the database is empty.
    - Accepts user queries in a loop and returns RAG-generated answers.
    """
    seed_urls = urls.seed_urls
    # TODO add args to adjust batch size and concurrency in FetchDocuments
    fetched_docs = FetchDocuments()
    chroma_client, collection = fetched_docs.create_chroma_db('chromadb_data')
    mod = AskModel(collection)

    if collection.count() == 0:
        print("No data found—starting crawl & ingest…")
        docs = await fetched_docs.crawl_and_clean(seed_urls)
        print(f"Fetched & cleaned {len(docs)} docs")
        passages = docs_to_passages(docs)
        print(f"Chunked into {len(passages)} passages")
        ingest_into_chromadb(passages, mod.embedder, chroma_client)
        print("Ingestion complete!")

    print("\nJapan Travel RAG (Phi-2) ready. Ask away!")
    while True:
        q = input("\nYour question (blank to exit): ").strip()
        if not q:
            break
        # TODO add these as args instead of hardcoding
        print("\n" + mod.answer_cached(q, temperature=0.3, top_p=0.85,
                                       repitition_penalty=1.2, max_new_tokens=256) + "\n")

if __name__ == "__main__":
    asyncio.run(main())
