import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
import torch
import mlflow
import os
import argparse

def compute_tfidf(input_file):
    with mlflow.start_run(run_name=f"TFIDF_{os.path.basename(input_file)}"):
        df = pd.read_csv(input_file)
        df["CleanedTitle"] = df["CleanedTitle"].fillna("").astype(str)

        max_features = 1000
        mlflow.log_param("model_type", "TF-IDF")
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("input_file", input_file)
        mlflow.log_metric("dataset_size", len(df))

        vectorizer = TfidfVectorizer(max_features=max_features)
        tfidf_matrix = vectorizer.fit_transform(df["CleanedTitle"])

        mlflow.log_metric("vocabulary_size", len(vectorizer.vocabulary_))
        mlflow.log_metric("feature_matrix_shape", tfidf_matrix.shape[1])

        feature_names = vectorizer.get_feature_names_out()
        top_features = ", ".join(feature_names[:20])
        mlflow.set_tag("top_features", top_features)

        print("TF-IDF Shape:", tfidf_matrix.shape)

        vocab_file = f"tfidf_vocab_{os.path.basename(input_file)}.txt"
        with open(vocab_file, 'w') as f:
            for word, idx in sorted(vectorizer.vocabulary_.items(), key=lambda x: x[1]):
                f.write(f"{word}: {idx}\n")
        mlflow.log_artifact(vocab_file)

def compute_bert_embeddings(input_file, batch_size=16):
    with mlflow.start_run(run_name=f"BERT_{os.path.basename(input_file)}"):
        df = pd.read_csv(input_file)
        df["CleanedTitle"] = df["CleanedTitle"].fillna("").astype(str)

        model_name = "bert-base-uncased"
        mlflow.log_param("model_type", "BERT")
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("input_file", input_file)
        mlflow.log_param("dataset_size", len(df))

        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
        model.eval()

        # Log model config
        for k, v in model.config.to_dict().items():
            if isinstance(v, (int, float, bool, str)):
                mlflow.log_param(f"bert_config_{k}", v)

        titles = df["CleanedTitle"].tolist()
        # Tokenization metrics
        encodings = tokenizer(titles, padding=True, truncation=True, return_tensors="pt")
        avg_token_length = np.mean([len(ids) for ids in encodings["input_ids"]])
        mlflow.log_metric("avg_token_length", avg_token_length)

        all_embeddings = []
        # Process in mini‑batches
        for i in range(0, len(titles), batch_size):
            batch_inputs = {k: t[i : i + batch_size] for k, t in encodings.items()}
            with torch.no_grad():
                outs = model(**batch_inputs)
            batch_emb = outs.last_hidden_state.mean(dim=1)  # [batch, hidden]
            all_embeddings.append(batch_emb)

        embeddings = torch.cat(all_embeddings, dim=0)  # [N, hidden]

        mlflow.log_metric("embedding_dimension", embeddings.shape[1])
        mlflow.log_metric("embedding_mean", float(embeddings.mean()))
        mlflow.log_metric("embedding_std", float(embeddings.std()))

        print("BERT Embeddings Shape:", embeddings.shape)

        sample_idx = min(5, embeddings.size(0))
        sample_file = f"bert_sample_embeddings_{os.path.basename(input_file)}.txt"
        with open(sample_file, 'w') as f:
            for j in range(sample_idx):
                f.write(f"Title: {titles[j][:50]}...\n")
                f.write(f"Embedding (first 10 dims): {embeddings[j][:10].tolist()}\n\n")
        mlflow.log_artifact(sample_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute TF-IDF and optional BERT embeddings")
    parser.add_argument("--input", type=str, required=True, help="Cleaned CSV file")
    parser.add_argument("--use_bert", action="store_true", help="Compute BERT embeddings")
    parser.add_argument("--bert_batch_size", type=int, default=16, help="BERT mini-batch size")
    args = parser.parse_args()

    # MLflow setup
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Text_Representation")

    print("Computing TF-IDF...")
    compute_tfidf(args.input)

    if args.use_bert:
        print("Computing BERT embeddings...")
        compute_bert_embeddings(args.input, batch_size=args.bert_batch_size)

    print("\nDone. Run mlflow ui to inspect results.")