import httpx
import asyncio
import chromadb
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE
from chromadb.errors import NotFoundError

from docs.utils import extract_text

class FetchDocuments:
    """
    Handles asynchronous fetching and cleaning of webpages, as well as ChromaDB initialization.

    Attributes:
        concurrency (int): Max concurrent requests.
        batch_size (int): Number of URLs fetched per batch.
    """
    def __init__(self, concurrency: int = 20, batch_size: int = 100):
        """Initialize fetcher with concurrency and batch size."""
        self.concurrency = concurrency
        self.batch_size = batch_size

    @staticmethod
    async def fetch_page(client: httpx.AsyncClient, url: str) -> str:
        """
        Asynchronously fetch HTML content from a URL.

        Args:
            client (httpx.AsyncClient): The HTTP client instance.
            url (str): The webpage URL.

        Returns:
            str: Raw HTML of the page.
        """
        r = await client.get(url, timeout=10.0)
        r.raise_for_status()
        return r.text

    async def crawl_and_clean(self, urls: list[str]) -> list[dict]:
        """
        Crawl and clean a list of webpages concurrently.

        Args:
            urls (list[str]): List of URLs to scrape.

        Returns:
            list[dict]: Cleaned documents with 'url' and 'text' keys.
        """
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
        """
        Create or load a persistent ChromaDB collection.

        Args:
            path (str): Filesystem path for database storage.

        Returns:
            tuple: (chroma_client, collection) instances.
        """
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