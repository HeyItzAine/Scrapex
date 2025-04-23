import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
import torch
import mlflow
import os

def compute_tfidf(input_file):
    with mlflow.start_run(run_name=f"TFIDF_{os.path.basename(input_file)}"):
        df = pd.read_csv(input_file)
        
        # Set and log parameters
        max_features = 1000
        mlflow.log_param("model_type", "TF-IDF")
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("input_file", input_file)
        mlflow.log_metric("dataset_size", len(df))
        
        vectorizer = TfidfVectorizer(max_features=max_features)
        tfidf_matrix = vectorizer.fit_transform(df["CleanedTitle"])
        
        # Log metrics
        mlflow.log_metric("vocabulary_size", len(vectorizer.vocabulary_))
        mlflow.log_metric("feature_matrix_shape", tfidf_matrix.shape[1])
        
        # Log feature names
        feature_names = vectorizer.get_feature_names_out()
        top_features = ", ".join(feature_names[:20])
        mlflow.set_tag("top_features", top_features)
        
        print("TF-IDF Shape:", tfidf_matrix.shape)
        
        # Save and log the vocabulary as artifact
        vocab_file = f"tfidf_vocab_{os.path.basename(input_file)}.txt"
        with open(vocab_file, 'w') as f:
            for word, idx in sorted(vectorizer.vocabulary_.items(), key=lambda x: x[1]):
                f.write(f"{word}: {idx}\n")
        mlflow.log_artifact(vocab_file)

def compute_bert_embeddings(input_file):
    with mlflow.start_run(run_name=f"BERT_{os.path.basename(input_file)}"):
        df = pd.read_csv(input_file)
        
        # Set and log parameters
        model_name = "bert-base-uncased"
        mlflow.log_param("model_type", "BERT")
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("input_file", input_file)
        mlflow.log_metric("dataset_size", len(df))
        
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
        
        # Log model configuration
        model_config = model.config.to_dict()
        for key, value in model_config.items():
            if isinstance(value, (int, float, str, bool)):
                mlflow.log_param(f"bert_config_{key}", value)
        
        titles = df["CleanedTitle"].tolist()
        inputs = tokenizer(titles, padding=True, truncation=True, return_tensors="pt")
        
        # Log tokenization metrics
        avg_token_length = np.mean([len(ids) for ids in inputs['input_ids']])
        mlflow.log_metric("avg_token_length", avg_token_length)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        embeddings = outputs.last_hidden_state.mean(dim=1)
        
        # Log embedding statistics
        mlflow.log_metric("embedding_dimension", embeddings.shape[1])
        mlflow.log_metric("embedding_mean", float(embeddings.mean()))
        mlflow.log_metric("embedding_std", float(embeddings.std()))
        
        print("BERT Embeddings Shape:", embeddings.shape)
        
        # Save and log sample embeddings
        sample_idx = min(5, len(embeddings))
        sample_file = f"bert_sample_embeddings_{os.path.basename(input_file)}.txt"
        with open(sample_file, 'w') as f:
            for i in range(sample_idx):
                f.write(f"Title: {titles[i][:50]}...\n")
                f.write(f"Embedding (first 10 dimensions): {embeddings[i][:10].tolist()}\n\n")
        mlflow.log_artifact(sample_file)

if __name__ == "__main__":
    # Set up MLflow tracking
    mlflow.set_tracking_uri("file:./mlruns")
    
    print("Computing TF-IDF of google scholar titles...")
    compute_tfidf("Data/research_titles_cleaned.csv")
    compute_bert_embeddings("Data/research_titles_cleaned.csv")
    print("Computing TF-IDF of semantic titles...")
    compute_tfidf("Data/semantic_titles_cleaned.csv")
    compute_bert_embeddings("Data/semantic_titles_cleaned.csv")
    
    print("\nMLflow tracking information:")
    print(f"- Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"- View the experiments at: http://localhost:5000 (after starting MLflow UI)")
