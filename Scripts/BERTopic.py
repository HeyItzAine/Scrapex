import pandas as pd
import sys
import os
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
    topic_model = BERTopic()  # Adjust parameters if needed
    
    topics, _ = topic_model.fit_transform(df['Content'].tolist())
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

def main():
    if len(sys.argv) != 3:
        print("Usage: python BERTopic.py <dataset_path> <topic_query>")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    query = sys.argv[2]
    base_filename = os.path.splitext(os.path.basename(dataset_path))[0]
    
    # Create a folder with base filename + "BERTopic" in the current working directory
    output_folder = os.path.join(os.getcwd(), f"{base_filename}_BERTopic")
    os.makedirs(output_folder, exist_ok=True)
    
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"Error: The file {dataset_path} was not found.")
        sys.exit(1)
    
    topic_model, topics = topic_modeling(df)
    df['Topic'] = topics
    
    # Save the BERTopic model to disk
    model_output_path = os.path.join(output_folder, f"{base_filename}_bertopic_model")
    topic_model.save(model_output_path)
    print(f"BERTopic model saved as: {model_output_path}")
    
    # Generate topic names (human-readable)
    topic_info = topic_model.get_topic_info()
    topic_name_map = {row["Topic"]: row["Name"] for _, row in topic_info.iterrows()}
    df["TopicName"] = df["Topic"].map(topic_name_map)
    
    # Show topics briefly
    for topic_num in set(topics):
        if topic_num != -1:
            print(f"\n🧠 Topic {topic_num}:")
            print(topic_model.get_topic(topic_num))
    
    # Save full output with topic names
    full_output = os.path.join(output_folder, f"{base_filename}_with_topics.csv")
    df.to_csv(full_output, index=False)
    print(f"\n✅ Topic modeling complete. Full output saved as: {full_output}")
    
    # Save filtered by query
    save_titles_by_query(df, query, base_filename, output_folder)
    
    # Optional: visualize topics
    try:
        topic_model.visualize_topics().show()
    except Exception as e:
        print("⚠️ Visualization failed (likely headless environment).")
    
if __name__ == "__main__":
    main()
