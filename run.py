from src import ArtifactLoader, Config, DualTierClassifier, HybridSearchEngine


def main():
    cfg = Config()
    loader = ArtifactLoader(cfg)
    search_engine = HybridSearchEngine(loader)
    classifier = DualTierClassifier(loader)

    sample_title = "End-to-End Object Detection with Transformers"
    sample_abstract = (
        "We present a new method that views object detection as a direct set "
        "prediction problem, utilizing a transformer encoder-decoder architecture."
    )

    print(f"QUERY PAPER : {sample_title}")

    print("\nTOP SIMILAR RESEARCH PAPERS:")
    recommendations = search_engine.search(sample_title, top_k=5)
    for rank, (paper_title, score) in enumerate(recommendations, 1):
        print(f"  {rank}. [Score: {score:+.2f}] {paper_title}")

    print("\nPREDICTED SUBJECT CATEGORIES:")
    categories = classifier.predict(sample_abstract)
    if not categories:
        print("  No category crossed the confidence threshold.")
    else:
        for cat in categories:
            print(f"  [{cat['code']}] {cat['name']}  | Conf:{cat['conf']:5.1f}% | {cat['tier']}")


if __name__ == "__main__":
    main()