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

    def __init__(self):
        self.model_name = "microsoft/phi-2"

    @cached_property
    def embedder(self):
        return SentenceTransformer("all-MiniLM-L6-v2")

    @cached_property
    def cross_encoder(self):
        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    @cached_property
    def tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

    def load_model(self):
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