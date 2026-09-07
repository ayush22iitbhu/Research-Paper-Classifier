from src import ArtifactLoader, DualTierClassifier, HybridSearchEngine

def main():
    # 1. Load artifacts
    loader = ArtifactLoader()

    # 2. Instantiate engines
    classifier = DualTierClassifier(loader)
    search_engine = HybridSearchEngine(loader)

    # 3. Run evaluation query
    sample_title = "End-to-End Object Detection with Transformers"
    sample_abstract = (
        "We present a new method that views object detection as a direct set "
        "prediction problem, utilizing a transformer encoder-decoder architecture."
    )

    print("=" * 70)
    print(f"QUERY TITLE: {sample_title}")
    print("=" * 70)

    # Subject Tagging
    categories = classifier.predict(sample_abstract)
    print("\n[PREDICTED SUBJECT CATEGORIES]")
    for cat in categories:
        print(f"  * [{cat['code']}] {cat['name']} ({cat['conf']} | {cat['tier']})")

    # Similar Paper Retrieval
    recommendations = search_engine.search(sample_title, top_k=5)
    print("\n[RECOMMENDED PAPERS]")
    for rank, (title, score) in enumerate(recommendations, 1):
        print(f"  {rank}. [Score: {score:+.2f}] {title}")
    print("=" * 70)

if __name__ == "__main__":
    main()