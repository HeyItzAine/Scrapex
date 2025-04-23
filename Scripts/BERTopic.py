import pandas as pd
import sys
import os
import mlflow
from bertopic import BERTopic

# Function to perform topic modeling with BERTopic
def topic_modeling(df):
    # Clean the data by filling missing values for the title
    df['CleanedTitle'] = df['CleanedTitle'].fillna('')
    
    # Check if 'CleanedAbstract' exists; if so, process it, else use title only
    if 'CleanedAbstract' in df.columns:
        df['CleanedAbstract'] = df['CleanedAbstract'].fillna('')
        df['Content'] = df['CleanedTitle'] + ' ' + df['CleanedAbstract']
    else:
        df['Content'] = df['CleanedTitle']
    
    print("✨ Creating new BERTopic model...")
    
    # Initialize BERTopic with parameters we want to track
    n_gram_range = (1, 2)
    min_topic_size = 10
    nr_topics = "auto"
    
    # Log parameters with MLflow
    mlflow.log_params({
        "n_gram_range": str(n_gram_range),
        "min_topic_size": min_topic_size,
        "nr_topics": nr_topics
    })
    
    topic_model = BERTopic(n_gram_range=n_gram_range, min_topic_size=min_topic_size, nr_topics=nr_topics)
    
    topics, probs = topic_model.fit_transform(df['Content'].tolist())
    
    # Log metrics with MLflow
    topic_info = topic_model.get_topic_info()
    num_topics = len([t for t in topic_info['Topic'] if t != -1])
    mlflow.log_metric("number_of_topics", num_topics)
    
    # Try to compute topic coherence if possible
    try:
        coherence = topic_model.calculate_probabilities(df['Content'].tolist()).mean()
        mlflow.log_metric("topic_coherence", coherence)
    except:
        print("⚠️ Could not calculate topic coherence")
    
    return topic_model, topics

# Save titles with topics (filtered or all)
def save_titles_by_query(df, query, base_filename, output_folder):
    if query == "all":
        filtered_df = df
    else:
        filtered_df = df[df['Topic'] == int(query)]
    
    output_file = os.path.join(output_folder, f"{base_filename}_topics_{query}_titles.csv")
    filtered_df[['CleanedTitle', 'TopicName']].to_csv(output_file, index=False)
    print(f"📄 CSV for Topic {query} saved as: {output_file}")
    
    # Log the output file as an artifact with MLflow
    mlflow.log_artifact(output_file)

def main():
    # Set up MLflow tracking
    # Use the local directory for tracking
    mlflow.set_tracking_uri("file:./mlruns")
    
    if len(sys.argv) != 3:
        print("Usage: python BERTopic.py <dataset_path> <topic_query>")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    query = sys.argv[2]
    base_filename = os.path.splitext(os.path.basename(dataset_path))[0]
    
    # Start an MLflow run with the dataset name as the run name
    with mlflow.start_run(run_name=f"BERTopic_{base_filename}"):
        # Log the input dataset path
        mlflow.log_param("dataset_path", dataset_path)
        mlflow.log_param("topic_query", query)
        
        # Create a folder with base filename + "BERTopic" in the current working directory
        output_folder = os.path.join(os.getcwd(), f"{base_filename}_BERTopic")
        os.makedirs(output_folder, exist_ok=True)
        
        try:
            df = pd.read_csv(dataset_path)
            mlflow.log_metric("dataset_size", len(df))
        except FileNotFoundError:
            print(f"Error: The file {dataset_path} was not found.")
            sys.exit(1)
        
        topic_model, topics = topic_modeling(df)
        df['Topic'] = topics
        
        # Save the BERTopic model to disk
        model_output_path = os.path.join(output_folder, f"{base_filename}_bertopic_model")
        topic_model.save(model_output_path)
        print(f"BERTopic model saved as: {model_output_path}")
        
        # Log the model as an artifact
        mlflow.log_artifact(model_output_path)
        
        # Generate topic names (human-readable)
        topic_info = topic_model.get_topic_info()
        topic_name_map = {row["Topic"]: row["Name"] for _, row in topic_info.iterrows()}
        df["TopicName"] = df["Topic"].map(topic_name_map)
        
        # Log topic distribution as metrics
        for topic_num in set(topics):
            if topic_num != -1:  # Skip outlier topic
                topic_count = sum(1 for t in topics if t == topic_num)
                topic_percentage = (topic_count / len(topics)) * 100
                mlflow.log_metric(f"topic_{topic_num}_percentage", topic_percentage)
                
                print(f"\n🧠 Topic {topic_num}:")
                topic_words = topic_model.get_topic(topic_num)
                print(topic_words)
                
                # Log top words for each topic
                top_words = ", ".join([word for word, _ in topic_words[:5]])
                mlflow.set_tag(f"topic_{topic_num}_top_words", top_words)
        
        # Save full output with topic names
        full_output = os.path.join(output_folder, f"{base_filename}_with_topics.csv")
        df.to_csv(full_output, index=False)
        print(f"\n✅ Topic modeling complete. Full output saved as: {full_output}")
        
        # Log the full output as an artifact
        mlflow.log_artifact(full_output)
        
        # Save filtered by query
        save_titles_by_query(df, query, base_filename, output_folder)
        
        # Optional: visualize topics and save for MLflow
        try:
            visualization = topic_model.visualize_topics()
            vis_path = os.path.join(output_folder, f"{base_filename}_topic_visualization.html")
            visualization.write_html(vis_path)
            mlflow.log_artifact(vis_path)
            
            # Try to show interactive visualization if not in headless environment
            visualization.show()
        except Exception as e:
            print(f"⚠️ Visualization error: {e}")
    
    print("MLflow tracking information:")
    print(f"- Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"- View the experiment at: http://localhost:5000 (after starting MLflow UI)")
    
if __name__ == "__main__":
    main()
