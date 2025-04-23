import os
import mlflow
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def check_data_exists():
    """Check if required data files exist"""
    files_needed = [
        "../Data/research_titles_cleaned.csv",
        "../Data/semantic_titles_cleaned.csv"
    ]
    missing = [f for f in files_needed if not os.path.exists(f)]
    if missing:
        print(f"Missing data files: {missing}")
        print("Please ensure the cleaned data files exist in the Data directory.")
        return False
    return True

def preprocess_data(data_file):
    """Load and preprocess the data"""
    try:
        df = pd.read_csv(data_file)
        print(f"Loaded {len(df)} records from {data_file}")
        return df
    except Exception as e:
        print(f"Error loading {data_file}: {e}")
        return None

def visualize_clusters(X_reduced, kmeans, dataset_name):
    """Create and save cluster visualization"""
    plt.figure(figsize=(10, 8))
    
    # Get a color palette based on the number of clusters
    palette = sns.color_palette("hls", kmeans.n_clusters)
    
    # Plot the clusters
    scatter = plt.scatter(
        X_reduced[:, 0], X_reduced[:, 1], 
        c=[palette[x] for x in kmeans.labels_],
        alpha=0.7
    )
    
    # Add centroids
    centroids = kmeans.cluster_centers_
    plt.scatter(
        centroids[:, 0], centroids[:, 1],
        marker='x', s=169, linewidths=3,
        color='black', zorder=10
    )
    
    plt.title(f'KMeans Clustering for {dataset_name}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    
    # Save figure - use absolute path to parent directory
    filename = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                           f"clusters_{dataset_name}.png")
    plt.savefig(filename)
    return filename

def analyze_dataset(data_file, experiment_name):
    """Run analysis on a dataset with MLflow tracking"""
    dataset_name = os.path.basename(data_file).replace(".csv", "")
    
    with mlflow.start_run(run_name=f"analysis_{dataset_name}"):
        # Log dataset info
        mlflow.log_param("dataset", dataset_name)
        mlflow.log_param("data_file", data_file)
        
        # Load and preprocess data
        df = preprocess_data(data_file)
        if df is None:
            return
        
        # Log data metrics
        mlflow.log_metric("n_records", len(df))
        
        # Extract titles for clustering
        titles = df["CleanedTitle"].fillna("").tolist()
        
        # Log sample titles
        sample_titles = titles[:5]
        for i, title in enumerate(sample_titles):
            mlflow.set_tag(f"sample_title_{i+1}", title)
        
        # Configure TF-IDF vectorizer
        n_features = 1000
        mlflow.log_param("n_features", n_features)
        
        # Create TF-IDF features
        vectorizer = TfidfVectorizer(
            max_features=n_features,
            stop_words='english',
            use_idf=True
        )
        X = vectorizer.fit_transform(titles)
        
        # Log vectorizer info
        mlflow.log_metric("vocab_size", len(vectorizer.vocabulary_))
        mlflow.log_metric("feature_matrix_shape", X.shape[1])
        
        # Save and log top terms
        terms = vectorizer.get_feature_names_out()
        top_terms = ", ".join(terms[:20])
        mlflow.set_tag("top_terms", top_terms)
        
        # Dimensionality reduction for visualization and clustering
        n_components = 2
        svd = TruncatedSVD(n_components)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)
        
        X_reduced = lsa.fit_transform(X)
        mlflow.log_metric("explained_variance", svd.explained_variance_ratio_.sum())
        mlflow.log_param("n_dimensions", n_components)
        
        # Try different cluster counts and find the best
        best_score = -1
        best_k = 2
        scores = []
        
        # Only try clustering if we have enough data
        min_k = 2
        max_k = min(10, len(titles) // 10)  # Don't try too many clusters for small datasets
        
        for k in range(min_k, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_reduced)
            
            # Skip silhouette calculation if only one sample in cluster
            if len(set(kmeans.labels_)) < k:
                continue 
                
            try:
                score = silhouette_score(X_reduced, kmeans.labels_)
                scores.append((k, score))
                mlflow.log_metric(f"silhouette_score_k{k}", score)
                
                if score > best_score:
                    best_score = score
                    best_k = k
            except:
                print(f"Couldn't calculate silhouette score for k={k}")
        
        # Log best cluster count
        mlflow.log_param("best_k", best_k)
        mlflow.log_metric("best_silhouette_score", best_score)
        
        # Create final clustering with best k
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        kmeans.fit(X_reduced)
        cluster_labels = kmeans.labels_
        
        # Add cluster labels to dataframe
        df["Cluster"] = cluster_labels
        
        # Log cluster distribution
        for i in range(best_k):
            count = np.sum(cluster_labels == i)
            percentage = (count / len(cluster_labels)) * 100
            mlflow.log_metric(f"cluster_{i}_size", count)
            mlflow.log_metric(f"cluster_{i}_percentage", percentage)
        
        # Save and log cluster visualization
        viz_file = visualize_clusters(X_reduced, kmeans, dataset_name)
        mlflow.log_artifact(viz_file)
        
        # Create and log cluster samples CSV - use absolute path to parent directory
        results_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                   f"cluster_results_{dataset_name}.csv")
        df[["CleanedTitle", "Cluster"]].to_csv(results_file, index=False)
        mlflow.log_artifact(results_file)
        
        # Clean up temporary files
        os.remove(viz_file)
        os.remove(results_file)
        
        print(f"Analysis complete for {dataset_name} with {best_k} clusters")
        return best_k, best_score

def main():
    # Check if data exists
    if not check_data_exists():
        return
    
    # Set up MLflow - point to mlruns in root directory
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mlflow.set_tracking_uri(f"file:{os.path.join(root_dir, 'mlruns')}")
    experiment_name = f"ScrapexML_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.set_experiment(experiment_name)
    
    print(f"Starting experiment: {experiment_name}")
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    
    # Run analysis on research titles
    print("\nAnalyzing research titles...")
    research_k, research_score = analyze_dataset("../Data/research_titles_cleaned.csv", experiment_name)
    
    # Run analysis on semantic titles
    print("\nAnalyzing semantic titles...")
    semantic_k, semantic_score = analyze_dataset("../Data/semantic_titles_cleaned.csv", experiment_name)
    
    # Log comparison
    with mlflow.start_run(run_name="comparison"):
        mlflow.log_param("research_best_k", research_k)
        mlflow.log_param("semantic_best_k", semantic_k)
        mlflow.log_metric("research_silhouette", research_score)
        mlflow.log_metric("semantic_silhouette", semantic_score)
        
        if research_score > semantic_score:
            conclusion = "Research titles clustering performed better"
        else:
            conclusion = "Semantic titles clustering performed better"
        
        mlflow.set_tag("conclusion", conclusion)
    
    print("\nExperiment complete!")
    print(f"View results at: http://localhost:5000/#/experiments")

if __name__ == "__main__":
    main() 