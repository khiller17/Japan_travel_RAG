from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import chromadb

def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "header", "footer", "nav", "form"]):
        tag.decompose()

    # Convert <li> elements into readable bullet points
    for li in soup.find_all("li"):
        li.insert_before(soup.new_tag("p"))
        li.string = f"- {li.get_text(strip=True)}"

    main = soup.find("main") or soup.find("article") or soup.find(attrs={"role": "main"})
    text_source = main or soup
    return "\n".join(p.get_text(strip=True) for p in text_source.find_all("p"))

def chunk_text(text: str, max_words: int = 500, overlap: int = 50):
    words = text.split()
    start = 0
    while start < len(words):
        end = min(len(words), start + max_words)
        yield " ".join(words[start:end])
        start += max_words - overlap

def docs_to_passages(docs: list[dict]) -> list[dict]:
    passages = []
    for doc in docs:
        for chunk in chunk_text(doc["text"]):
            if len(chunk.split()) > 50:
                passages.append({"url": doc["url"], "text": chunk})
    return passages

def ingest_into_chromadb(passages: list[dict], embedder: SentenceTransformer,
                         client: chromadb.Client, collection_name: str = "japan_travel",
                         batch_size: int = 64):
    try:
        col = client.get_collection(collection_name)
    except ValueError:
        col = client.create_collection(collection_name)

    texts = [p["text"] for p in passages]
    metadatas = [{"url": p["url"]} for p in passages]
    ids = [str(i) for i in range(len(passages))]

    for i in range(0, len(texts), batch_size):
        bt = texts[i:i + batch_size]
        bm = metadatas[i:i + batch_size]
        bi = ids[i:i + batch_size]
        embs = embedder.encode(bt, batch_size=batch_size,
                               convert_to_numpy=True,
                               show_progress_bar=False).tolist()

        col.add(documents=bt, metadatas=bm, ids=bi, embeddings=embs)