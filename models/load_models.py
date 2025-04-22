import os
from functools import cached_property

import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM
# Set threading limits for performance
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
torch.set_num_threads(8)

class LoadModels:
    """
    Loads and caches model components for embedding and generation.

    - SentenceTransformer for dense retrieval.
    - CrossEncoder for reranking.
    - Phi-2 (quantized) as the local LLM.
    """

    def __init__(self):
        self.model_name = "microsoft/phi-2"

    @cached_property
    def embedder(self):
        """Load the embedding model."""
        return SentenceTransformer("all-MiniLM-L6-v2")

    @cached_property
    def cross_encoder(self):
        """Load the cross encoder for re-ranking"""
        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    @cached_property
    def tokenizer(self):
        """Load tokenizer for Phi-2."""
        return AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

    def load_model(self):
        """
        Load Phi-2 model from file or download and quantize it.

        Returns:
            torch.nn.Module: The quantized LLM model.
        """
        try:
            model = torch.load("models/phi_quantized.pt", weights_only=False)
            print("Loading model from saved file")
        except FileNotFoundError:
            print("Saved model not found. Loading, quantizing, and saving model...this takes a few minutes")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32
                # low_cpu_mem_usage=True
            )
            model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            torch.save(model, "models/phi_quantized.pt")
        return model