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
