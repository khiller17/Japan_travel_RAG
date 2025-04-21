# 🇯🇵 Japan Travel RAG

This project is a Retrieval-Augmented Generation (RAG) application designed to answer travel-related questions about Japan. It uses a local, quantized language model and a curated set of Japanese food blogs and restaurant review sites to provide relevant, conversational answers—without requiring GPU support.

> ⚠️ **Work in Progress**:  
> The current version focuses primarily on **food and restaurant recommendations** in Japan. It uses a limited set of URLs for data retrieval. Support for additional topics and broader content coverage is planned for future updates. Future updates will also make the answers more concise.

---

## Features

- **Retrieval-Augmented Generation (RAG):** Combines document retrieval and generative language models for contextual responses.
- **Local Inference:** Uses a quantized LLM (microsoft phi-2) to run efficiently on CPU-only machines.
- **Custom Web Scraper:** Extracts and chunks food-related content from Japanese travel and culinary blogs.
- **Fast Search:** Uses ChromaDB as a vector database for quick document retrieval.
- **Answer Tuning:** Uses prompt engineering to discourage generic, non-informative language and guide the model toward concise, helpful answers.

---

## Technologies

- Python 3.10+
- [SentenceTransformers](https://www.sbert.net/)
- [Transformers (HuggingFace)](https://huggingface.co/docs/transformers/)
- [ChromaDB](https://docs.trychroma.com/)
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/)
- Quantized LLMs (`Phi-2`)

---

## Project Structure

Japan_travel_RAG/
├── docs/  
│   ├── load_docs.py          # Async scraping and text chunking, chroma DB
│   ├── utils.py              # utils for chunking and chroma ingestion
│   ├── urls.py      
├── models/
│   ├── load_models.py        # load tokenizer, embedder, load and quantize model
│   ├── query_model.py        # prompt and query quantized model
├── main.py                   # main entrypoint
├── .gitignore
└── environment.yml

## Quickstart

1. **Clone the repo**
   ```bash
   git clone https://github.com/khiller17/Japan_travel_RAG.git
   cd Japan_travel_RAG

2. **Create virtual environment**
   conda env create -f environment.yml
   conda activate rag_demo

3. **Run main.py**

## Example query

Your question (blank to exit): What’s a good ramen shop in Shibuya?

Answer in one paragraph: Ichiran Shibuya is a popular ramen spot known for its tonkotsu broth and solo dining booths. It's centrally located and ideal for travelers looking for a quick, high-quality meal.
