"""
TRAINING: Extract Embeddings from Multiple Models
Extracts embeddings from 6 different Hugging Face models for comparison

IEEE References:
- DepRoBERTa: Poswiata et al. 2022 (LT-EDI Workshop)
- BERT-RF-LR: Abbas et al. 2024 (DOI: 10.1109/ACCESS.2024.3387695)
- Mental health models: Ji et al. 2022, Shen et al. 2023

Usage:
    cd C:/Users/Disha/OneDrive/Desktop/Transformer-mental-risk-2
    python backend/training/extract_embeddings.py
"""

import sys
import os

# Add project root to path for imports
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
import time
from collections import Counter

# Import your services
from backend.utils.textPreprocessor import RedditPreprocessor
from backend.app.services.embeddings_service import EmbeddingService


# ============= MODELS TO COMPARE =============
MODELS_TO_COMPARE = {
    # "deproberta-large": {
    #     "name": "rafalposwiata/deproberta-large-depression",
    #     "batch_size": 4,  # Large model - small batch
    #     "desc": "DepRoBERTa Large (Depression Specialized)",
    #     "reference": "Poswiata et al. 2022",
    # },
    # "distilbert-sst2": {
    #     "name": "distilbert-base-uncased-finetuned-sst-2-english",
    #     "batch_size": 32,  # Small model - large batch
    #     "desc": "DistilBERT SST-2 (Sentiment Analysis)",
    #     "reference": "Sanh et al. 2019",
    # },
    # "bert-base": {
    #     "name": "bert-base-uncased",
    #     "batch_size": 16,
    #     "desc": "BERT Base (General Purpose)",
    #     "reference": "Devlin et al. 2019",
    # },
    # "roberta-base": {
    #     "name": "roberta-base",
    #     "batch_size": 16,
    #     "desc": "RoBERTa Base (General Purpose)",
    #     "reference": "Liu et al. 2019",
    # },
    "distilroberta": {
        "name": "distilroberta-base",
        "batch_size": 32,
        "desc": "DistilRoBERTa Base (Efficient RoBERTa)",
        "reference": "Sanh et al. 2019",
    },
}


def analyze_dataset(df):
    """Analyze class distribution"""
    print("\n" + "=" * 70)
    print("DATASET ANALYSIS")
    print("=" * 70)

    labels = df["labels"].values
    counter = Counter(labels)
    total = len(labels)

    label_names = {0: "not_depressed", 1: "moderate", 2: "severe"}

    print(f"\n📊 Total samples: {total:,}")
    print(f"\n📈 Class distribution:")
    for label in sorted(counter.keys()):
        count = counter[label]
        pct = (count / total) * 100
        name = label_names.get(label, f"class_{label}")
        print(f"  {name} (label {label}): {count:,} ({pct:.2f}%)")

    # Calculate imbalance
    max_class = max(counter.values())
    min_class = min(counter.values())
    imbalance_ratio = max_class / min_class

    print(f"\n⚖️ Imbalance ratio: {imbalance_ratio:.2f}x")
    if imbalance_ratio > 1.5:
        print("  ⚠️ CLASS IMBALANCE DETECTED!")
        print("  → SMOTE will be applied during model training")

    return imbalance_ratio


def extract_all_embeddings():
    """
    Main function: Extract embeddings from ALL models
    Saves to backend/models/embeddings/
    """

    print("\n" + "🤖" * 35)
    print("MULTI-MODEL EMBEDDING EXTRACTION")
    print("🤖" * 35)
    print(f"\nExtracting embeddings from {len(MODELS_TO_COMPARE)} models")
    print("This will take 2-3 hours...")

    # ========== STEP 1: LOAD DATA ==========
    print("\n" + "=" * 70)
    print("STEP 1: LOADING DATA")
    print("=" * 70)

    data_path = os.path.join(project_root, "data", "datasets", "combined_cleaned.csv")

    if not os.path.exists(data_path):
        print(f"❌ ERROR: {data_path} not found!")
        print("   Make sure combined_cleaned.csv exists in data/datasets/")
        return

    df = pd.read_csv(data_path)
    print(f"✓ Loaded dataset from: {data_path}")
    print(f"  Samples: {len(df):,}")
    print(f"  Columns: {df.columns.tolist()}")

    # Analyze class distribution
    imbalance_ratio = analyze_dataset(df)

    # ========== STEP 2: PREPROCESS ==========
    print("\n" + "=" * 70)
    print("STEP 2: PREPROCESSING")
    print("=" * 70)

    print("Creating preprocessor...")
    preprocessor = RedditPreprocessor()

    print(f"Preprocessing {len(df):,} texts...")
    start_time = time.time()

    texts_clean = []
    for text in df["text"].tolist():
        clean = preprocessor.clean(text)
        texts_clean.append(clean)

    preprocess_time = time.time() - start_time

    print(f"✓ Preprocessed {len(texts_clean):,} texts in {preprocess_time:.2f}s")
    print(
        f"  Average length: {np.mean([len(t.split()) for t in texts_clean]):.1f} words"
    )

    # Show example
    print(f"\n📝 Example:")
    print(f"  Original: {df['text'].iloc[0][:100]}...")
    print(f"  Cleaned:  {texts_clean[0][:100]}...")

    # Get labels
    labels = df["labels"].values

    # ========== STEP 3: CREATE EMBEDDINGS DIRECTORY ==========
    embeddings_dir = os.path.join(project_root, "backend", "models", "embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)
    print(f"\n📁 Embeddings will be saved to: {embeddings_dir}")

    # ========== STEP 4: EXTRACT EMBEDDINGS FROM ALL MODELS ==========
    print("\n" + "=" * 70)
    print("STEP 3: EXTRACTING EMBEDDINGS FROM ALL MODELS")
    print("=" * 70)

    results = []
    total_start_time = time.time()

    for idx, (model_key, config) in enumerate(MODELS_TO_COMPARE.items(), 1):
        print(f"\n{'='*70}")
        print(f"MODEL {idx}/{len(MODELS_TO_COMPARE)}: {config['desc']}")
        print(f"{'='*70}")
        print(f"  Name: {config['name']}")
        print(f"  Batch size: {config['batch_size']}")
        print(f"  Reference: {config['reference']}")

        model_start_time = time.time()

        try:
            # Create service for this model
            service = EmbeddingService(model_name=config["name"])

            # Extract embeddings
            print(f"\n  🔄 Extracting embeddings...")
            embeddings = service.extract_batch(
                texts_clean,
                batch_size=config["batch_size"],
                max_length=256,
                show_progress=True,
            )

            # Verify shape
            print(f"\n  ✓ Extracted embeddings")
            print(f"    Shape: {embeddings.shape}")
            print(f"    Expected: ({len(texts_clean)}, embedding_dim)")

            # Save embeddings using numpy
            emb_path = os.path.join(embeddings_dir, f"{model_key}_embeddings.npy")
            np.save(emb_path, embeddings)

            model_elapsed = time.time() - model_start_time

            print(f"\n  ✅ SUCCESS!")
            print(f"     File: {emb_path}")
            print(f"     Size: {embeddings.nbytes / 1024 / 1024:.2f} MB")
            print(f"     Time: {model_elapsed:.2f}s ({model_elapsed/60:.2f} min)")

            results.append(
                {
                    "model": model_key,
                    "description": config["desc"],
                    "shape": embeddings.shape,
                    "time_seconds": model_elapsed,
                    "file": emb_path,
                }
            )

            # Free memory before loading next model
            service.unload()

        except Exception as e:
            print(f"\n  ❌ ERROR: {str(e)}")
            import traceback

            traceback.print_exc()
            continue

    # # ========== STEP 5: SAVE LABELS ==========
    # print("\n" + "=" * 70)
    # print("STEP 4: SAVING LABELS")
    # print("=" * 70)

    # labels_path = os.path.join(embeddings_dir, "labels.npy")
    # np.save(labels_path, labels)
    # print(f"✓ Saved labels: {labels_path}")
    # print(f"  Shape: {labels.shape}")

    # # ========== STEP 6: SAVE METADATA ==========
    # print("\n" + "=" * 70)
    # print("STEP 5: SAVING METADATA")
    # print("=" * 70)

    # import json

    # metadata = {
    #     "total_samples": len(texts_clean),
    #     "num_classes": len(np.unique(labels)),
    #     "class_distribution": {int(k): int(v) for k, v in Counter(labels).items()},
    #     "imbalance_ratio": float(imbalance_ratio),
    #     "models_extracted": len(results),
    #     "extraction_date": time.strftime("%Y-%m-%d %H:%M:%S"),
    #     "models": [
    #         {
    #             "key": r["model"],
    #             "description": r["description"],
    #             "shape": list(r["shape"]),
    #             "time_seconds": r["time_seconds"],
    #         }
    #         for r in results
    #     ],
    # }

    # metadata_path = os.path.join(embeddings_dir, "metadata.json")
    # with open(metadata_path, "w") as f:
    #     json.dump(metadata, f, indent=2)

    # print(f"✓ Saved metadata: {metadata_path}")

    # # ========== FINAL SUMMARY ==========
    # total_time = time.time() - total_start_time

    # print("\n" + "=" * 70)
    # print("EXTRACTION COMPLETE - SUMMARY")
    # print("=" * 70 + "\n")

    # for result in results:
    #     print(f"✓ {result['model']}: {result['description']}")
    #     print(f"  Shape: {result['shape']}")
    #     print(
    #         f"  Time: {result['time_seconds']:.2f}s ({result['time_seconds']/60:.2f} min)"
    #     )
    #     print(f"  File: {result['file']}\n")

    # print(f"📊 Total samples: {len(texts_clean):,}")
    # print(f"🏷️ Labels: {labels_path}")
    # print(f"📋 Metadata: {metadata_path}")
    # print(f"⏱️ Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
    # print(f"💾 Saved to: {embeddings_dir}")

    # print("\n" + "✅" * 35)
    # print("ALL EMBEDDINGS EXTRACTED SUCCESSFULLY!")
    # print("✅" * 35)
    # print("\n📌 Next step:")
    # print("   Run: python backend/training/train_models.py")
    # print("   This will train RF+LR on all embeddings")

    # return results


if __name__ == "__main__":
    extract_all_embeddings()
