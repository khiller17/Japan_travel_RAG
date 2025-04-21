import httpx
import asyncio
import chromadb
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE
from chromadb.errors import NotFoundError

from docs.utils import extract_text

class FetchDocuments:
    def __init__(self, concurrency: int = 20, batch_size: int = 100):
        self.concurrency = concurrency
        self.batch_size = batch_size

    @staticmethod
    async def fetch_page(client: httpx.AsyncClient, url: str) -> str:
        r = await client.get(url, timeout=10.0)
        r.raise_for_status()
        return r.text

    async def crawl_and_clean(self, urls: list[str]) -> list[dict]:
        sem = asyncio.Semaphore(self.concurrency)
        results = []
        async with httpx.AsyncClient() as client:
            for i in range(0, len(urls), self.batch_size):
                batch = urls[i:i + self.batch_size]

                async def safe_fetch(u: str):
                    async with sem:
                        try:
                            html = await self.fetch_page(client, u)
                            return {"url": u, "text": extract_text(html)}
                        except Exception as e:
                            print(f"[crawl error] {u}: {e}")
                            return None

                tasks = [asyncio.create_task(safe_fetch(u)) for u in batch]
                for task in asyncio.as_completed(tasks):
                    res = await task
                    if res:
                        results.append(res)
        return results

    @staticmethod
    def create_chroma_db(path):
        chroma_client = chromadb.PersistentClient(
            path=path,
            settings=Settings(),
            tenant=DEFAULT_TENANT,
            database=DEFAULT_DATABASE,
        )

        try:
            collection = chroma_client.get_collection("japan_travel")
        except NotFoundError:
            collection = chroma_client.create_collection("japan_travel")
        return chroma_client, collection