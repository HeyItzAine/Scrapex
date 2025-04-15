import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
import torch

def compute_tfidf(input_file):
    df = pd.read_csv(input_file)
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(df["CleanedTitle"])

    print("TF-IDF Shape:", tfidf_matrix.shape)

def compute_bert_embeddings(input_file):
    df = pd.read_csv(input_file)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    titles = df["CleanedTitle"].tolist()
    inputs = tokenizer(titles, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state.mean(dim=1)
    print("BERT Embeddings Shape:", embeddings.shape)

if __name__ == "__main__":
    print("Computing TF-IDF of google scholar titles...")
    compute_tfidf("Data/research_titles_cleaned.csv")
    compute_bert_embeddings("Data/research_titles_cleaned.csv")
    print("Computing TF-IDF of semantic titles...")
    compute_tfidf("Data/semantic_titles_cleaned.csv")
    compute_bert_embeddings("Data/semantic_titles_cleaned.csv")
