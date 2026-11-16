from src import (
    Config,
    build_and_save_artifacts,
    evaluate_dual_tier_system,
    load_and_clean_data,
    prepare_head_splits,
    train_bi_encoder,
    train_mlp_head,
)


def main():
    cfg = Config()

    # Load & sanitize dataset
    arxiv_data, arxiv_taxonomy = load_and_clean_data(cfg)
    sentences = arxiv_data["titles"].astype(str).tolist()
    abstract_sentences = arxiv_data["abstracts"].astype(str).tolist()
    y_raw = arxiv_data["terms"].tolist()

    # Contrastive domain adaptation
    st_model = train_bi_encoder(arxiv_data, cfg)

    # Encode corpus
    title_embeddings = st_model.encode(sentences, normalize_embeddings=True, convert_to_numpy=True)
    abstract_embeddings = st_model.encode(abstract_sentences, convert_to_numpy=True)

    # Split & train supervised MLP head
    X_train, y_train_encoded, X_test, idx_test, mlb = prepare_head_splits(abstract_embeddings, y_raw, cfg)
    mlp_model = train_mlp_head(X_train, y_train_encoded, cfg)

    # Build & serialize indices + model weights
    rare_codes, rare_embeddings = build_and_save_artifacts(
        st_model, mlp_model, mlb, sentences, title_embeddings, arxiv_taxonomy, cfg
    )

    # Run held-out evaluation
    test_raw_terms = [y_raw[i] for i in idx_test]
    evaluate_dual_tier_system(mlp_model, mlb, rare_codes, rare_embeddings, X_test, test_raw_terms, cfg)


if __name__ == "__main__":
    main()