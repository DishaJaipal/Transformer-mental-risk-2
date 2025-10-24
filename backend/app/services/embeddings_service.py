"""
Embedding Service - Extracts embeddings from any Hugging Face transformer model
REUSABLE for Training (batch processing) & Production (single text)

IEEE References:
- BERT embeddings: Abbas et al. 2024 (DOI: 10.1109/ACCESS.2024.3387695)
- DepRoBERTa: Poswiata et al. 2022 (LT-EDI Workshop)
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import warnings

warnings.filterwarnings("ignore")


class EmbeddingService:
    """
    Flexible embedding extraction service
    Works with ANY Hugging Face transformer model
    """

    def __init__(self, model_name="rafalposwiata/deproberta-large-v1", device=None):
        """
        Initialize embedding service

        Args:
            model_name (str): Any Hugging Face model identifier
            device (str): 'cuda' or 'cpu' (auto-detects if None)
        """
        self.model_name = model_name

        # Force GPU detection
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                print(f"✓ EmbeddingService created")
                print(f"  Model: {model_name}")
                print(f"  🎮 GPU: {torch.cuda.get_device_name(0)}")
                print(f"  Device: {self.device}")
            else:
                self.device = torch.device("cpu")
                print(f"✓ EmbeddingService created")
                print(f"  Model: {model_name}")
                print(f"  ⚠️ Using CPU (no GPU detected)")
        else:
            self.device = torch.device(device)
            print(f"✓ EmbeddingService created")
            print(f"  Model: {model_name}")
            print(f"  Device: {self.device}")

        # Lazy loading
        self._tokenizer = None
        self._model = None
        self._is_loaded = False

        print(f"✓ EmbeddingService created")
        print(f"  Model: {model_name}")
        print(f"  Device: {self.device}")

    def _load_model(self):
        """Load model (called automatically on first use)"""
        if self._is_loaded:
            return

        print(f"\n🔄 Loading {self.model_name}...")
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
            self._is_loaded = True
            print(f"✓ Model loaded on {self.device}")
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise

    def extract_single(self, text, max_length=256):
        """
        Extract embedding for ONE text (for production)

        Args:
            text (str): Preprocessed text
            max_length (int): Max sequence length

        Returns:
            np.ndarray: Shape (1, embedding_dim)
        """
        if not self._is_loaded:
            self._load_model()

        with torch.no_grad():
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self._model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return embedding

    def extract_batch(self, texts, batch_size=16, max_length=256, show_progress=True):
        """
        Extract embeddings for MANY texts (for training)

        Args:
            texts (list): List of preprocessed texts
            batch_size (int): Batch size (4-8 for large models, 16-32 for small)
            max_length (int): Max sequence length
            show_progress (bool): Show progress bar

        Returns:
            np.ndarray: Shape (N, embedding_dim)
        """
        if not self._is_loaded:
            self._load_model()

        embeddings = []

        # Progress bar
        if show_progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(range(0, len(texts), batch_size), desc="  Extracting")
            except ImportError:
                iterator = range(0, len(texts), batch_size)
                print(f"  Processing {len(texts):,} texts...")
        else:
            iterator = range(0, len(texts), batch_size)

        with torch.no_grad():
            for i in iterator:
                batch = texts[i : i + batch_size]

                inputs = self._tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=max_length,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self._model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_embeddings)

        return np.vstack(embeddings)

    def get_embedding_dim(self):
        """Get embedding dimension"""
        if not self._is_loaded:
            self._load_model()
        return self._model.config.hidden_size

    def unload(self):
        """Free GPU memory (important when switching models!)"""
        if self._model is not None:
            del self._model
            del self._tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._model = None
            self._tokenizer = None
            self._is_loaded = False
            print(f"✓ Unloaded: {self.model_name}")


# Quick test
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("EMBEDDING SERVICE TEST")
    print("=" * 70)

    service = EmbeddingService("bert-base-uncased")

    # Test single
    text = "i feel sad"
    emb = service.extract_single(text)
    print(f"\nSingle: {emb.shape}")

    # Test batch
    texts = ["i feel sad", "i am happy", "i feel lonely"]
    embs = service.extract_batch(texts, batch_size=2, show_progress=False)
    print(f"Batch: {embs.shape}")

    # Test unload
    service.unload()

    print("\n✅ All tests passed!")
