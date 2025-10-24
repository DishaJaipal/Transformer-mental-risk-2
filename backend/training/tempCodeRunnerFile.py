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
