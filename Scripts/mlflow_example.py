import os
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import load_iris

def clean_name(name):
    """Clean feature names to be MLflow-compatible"""
    return name.replace(" ", "_").replace("(", "").replace(")", "")

def run_experiment(n_estimators, max_depth, random_state=42):
    """
    Train a Random Forest classifier on the Iris dataset with MLflow tracking
    """
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    # Clean feature names for MLflow
    clean_feature_names = [clean_name(name) for name in feature_names]
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"rf_n{n_estimators}_d{max_depth}"):
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("dataset", "iris")
        mlflow.log_param("model_type", "RandomForestClassifier")
        
        # Log dataset info
        mlflow.log_metric("dataset_size", len(X))
        mlflow.log_metric("n_features", X.shape[1])
        mlflow.log_metric("n_classes", len(np.unique(y)))
        
        # Train the model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        model.fit(X_train, y_train)
        
        # Make predictions and calculate metrics
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Log feature importances
        importances = model.feature_importances_
        for i, importance in enumerate(importances):
            mlflow.log_metric(f"importance_{clean_feature_names[i]}", importance)
        
        # Create and log a feature importance CSV - use absolute path to parent directory
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        importance_file = os.path.join(root_dir, "feature_importances.csv")
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        importance_df.to_csv(importance_file, index=False)
        mlflow.log_artifact(importance_file)
        os.remove(importance_file)  # Clean up
        
        # Save and log the model
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        # Create and log confusion matrix
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        # Save and log the confusion matrix - use absolute path to parent directory
        cm_file = os.path.join(root_dir, "confusion_matrix.png")
        plt.tight_layout()
        plt.savefig(cm_file)
        mlflow.log_artifact(cm_file)
        os.remove(cm_file)  # Clean up
        
        print(f"Run completed with accuracy: {accuracy:.4f}")
        return accuracy

def main():
    # Set up MLflow tracking - point to mlruns in root directory
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mlflow.set_tracking_uri(f"file:{os.path.join(root_dir, 'mlruns')}")
    mlflow.set_experiment("IrisClassification")
    
    print("MLflow tracking URI:", mlflow.get_tracking_uri())
    
    # Run experiments with different hyperparameters
    results = []
    
    # Different n_estimators
    for n_estimators in [10, 50, 100]:
        for max_depth in [None, 5, 10]:
            accuracy = run_experiment(n_estimators, max_depth)
            results.append({
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'accuracy': accuracy
            })
    
    # Display results
    results_df = pd.DataFrame(results)
    print("\nExperiment Results:")
    print(results_df.sort_values('accuracy', ascending=False))
    
    print("\nTo view the MLflow UI, run: mlflow ui")
    print("Then open http://localhost:5000 in your browser")

if __name__ == "__main__":
    main() 