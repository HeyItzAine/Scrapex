import os
import argparse
import mlflow
import subprocess
import webbrowser
from datetime import datetime

def setup_mlflow():
    """Set up MLflow tracking"""
    # Using local directory for tracking - point to mlruns in root directory
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mlflow.set_tracking_uri(f"file:{os.path.join(root_dir, 'mlruns')}")
    print(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")

def run_ml_pipeline(run_text_representation=True, run_bertopic=True, start_ui=True):
    """Run the ML pipeline with MLflow tracking"""
    experiment_name = f"ScrapexML_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.set_experiment(experiment_name)
    
    print(f"🚀 Starting experiment: {experiment_name}")
    
    # Run Text Representation
    if run_text_representation:
        print("\n📊 Running Text Representation models...")
        subprocess.run(["python", "./Text_Representation.py"])
    
    # Run BERTopic
    if run_bertopic:
        print("\n🧠 Running BERTopic models...")
        # Research titles
        subprocess.run([
            "python", "./BERTopic.py", 
            "../Data/research_titles_cleaned.csv", "all"
        ])
        
        # Semantic titles
        subprocess.run([
            "python", "./BERTopic.py", 
            "../Data/semantic_titles_cleaned.csv", "all"
        ])
    
    # Start MLflow UI in a separate process
    if start_ui:
        print("\n🌐 Starting MLflow UI at http://localhost:5000")
        print("(Press CTRL+C to stop the server when you're done)")
        
        # Try to open the UI in the default web browser
        try:
            webbrowser.open("http://localhost:5000")
        except:
            print("Could not open browser automatically. Please navigate to http://localhost:5000")
        
        # Start MLflow UI from the root directory
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.chdir(root_dir)  # Change to root directory before starting UI
        subprocess.run(["mlflow", "ui", "--port", "5000"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ML experiments with MLflow tracking")
    parser.add_argument("--skip-text", action="store_true", help="Skip Text Representation step")
    parser.add_argument("--skip-bertopic", action="store_true", help="Skip BERTopic step")
    parser.add_argument("--no-ui", action="store_true", help="Don't start MLflow UI")
    
    args = parser.parse_args()
    
    setup_mlflow()
    run_ml_pipeline(
        run_text_representation=not args.skip_text,
        run_bertopic=not args.skip_bertopic,
        start_ui=not args.no_ui
    ) 