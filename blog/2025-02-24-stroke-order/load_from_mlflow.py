import mlflow
import os
import argparse
from stable_baselines3 import PPO
import numpy as np
import torch
from dataset import StrokeOrderDataset
from train import StrokeOrderEnv

def load_model_from_mlflow(run_id, model_path="models/best_stroke_order_model.zip"):
    """
    Load a model artifact from MLflow.
    
    Args:
        run_id (str): The MLflow run ID to load the model from
        model_path (str): Path to the model within the artifacts
        
    Returns:
        model: The loaded model
    """
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:8090")
    
    # Get the artifact URI for the run
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    artifact_uri = run.info.artifact_uri
    
    # Construct the full path to the model
    local_path = os.path.join(artifact_uri, model_path)
    
    # If the path starts with file:, remove it
    if local_path.startswith("file:"):
        local_path = local_path[5:]
    
    print(f"Loading model from: {local_path}")
    
    # Load the model
    model = PPO.load(local_path)
    
    return model, run

def evaluate_model(model, num_samples=5):
    """
    Evaluate a loaded model on random samples.
    
    Args:
        model: The loaded model to evaluate
        num_samples (int): Number of samples to evaluate on
    """
    # Create environment
    env = StrokeOrderEnv()
    
    print("\nEvaluating model...")
    print("-" * 50)
    
    # Track overall statistics
    total_accuracy = 0
    
    for i in range(num_samples):
        # Reset environment to get a random character
        obs, _ = env.reset()
        done = False
        
        # Get character info
        sample = env.current_sample
        char_symbol = sample['character']
        
        print(f"\nSample {i+1}: Character '{char_symbol}'")
        print(f"Target strokes: {env.target_strokes}")
        
        # Track predicted strokes
        predicted_strokes = []
        
        while not done:
            # Prepare observation for model
            obs_dict = {
                'image': torch.tensor(np.expand_dims(obs['image'], axis=0), dtype=torch.float32),
                'stroke_history': torch.tensor(np.expand_dims(obs['stroke_history'], axis=0), dtype=torch.float32)
            }
            
            # Get model prediction
            action, _states = model.predict(obs_dict, deterministic=True)
            
            # Convert action to stroke name
            stroke_name = env.stroke_types[action[0]]
            predicted_strokes.append(stroke_name)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action[0])
            done = terminated or truncated
            
            if done:
                break
        
        print(f"Predicted strokes: {predicted_strokes}")
        
        # Calculate accuracy
        correct = 0
        for j, (pred, target) in enumerate(zip(predicted_strokes, env.target_strokes)):
            if pred == target:
                correct += 1
            else:
                break
                
        accuracy = correct / len(env.target_strokes) if len(env.target_strokes) > 0 else 0
        total_accuracy += accuracy
        print(f"Accuracy: {accuracy:.2%}")
        print("-" * 30)
    
    # Print overall statistics
    print("\nOverall Statistics:")
    print(f"Average accuracy: {(total_accuracy / num_samples):.2%}")

def list_mlflow_runs(experiment_id=0, max_runs=10):
    """
    List available MLflow runs.
    
    Args:
        experiment_id (int): The experiment ID to list runs from
        max_runs (int): Maximum number of runs to list
    """
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:8090")
    
    # Get the client
    client = mlflow.tracking.MlflowClient()
    
    # Get runs
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        max_results=max_runs,
        order_by=["attribute.start_time DESC"]
    )
    
    print("\nAvailable MLflow runs:")
    print("-" * 50)
    
    for i, run in enumerate(runs):
        run_id = run.info.run_id
        run_name = run.data.tags.get("mlflow.runName", "Unnamed run")
        start_time = run.info.start_time
        
        # Get metrics
        metrics = run.data.metrics
        mean_reward = metrics.get("eval/mean_reward", "N/A")
        success_rate = metrics.get("eval/success_rate", "N/A")
        
        print(f"{i+1}. Run ID: {run_id}")
        print(f"   Name: {run_name}")
        print(f"   Mean Reward: {mean_reward}")
        print(f"   Success Rate: {success_rate}")
        print("-" * 30)
    
    return runs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and evaluate a model from MLflow")
    parser.add_argument("--run-id", type=str, help="MLflow run ID to load model from")
    parser.add_argument("--model-path", type=str, default="models/best_stroke_order_model.zip", 
                        help="Path to model within the artifacts")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to evaluate on")
    parser.add_argument("--list-runs", action="store_true", help="List available MLflow runs")
    
    args = parser.parse_args()
    
    if args.list_runs:
        runs = list_mlflow_runs()
        
        # Ask user to select a run if run_id is not provided
        if not args.run_id:
            run_index = input("\nEnter the number of the run to load (or 'q' to quit): ")
            if run_index.lower() == 'q':
                exit(0)
            
            try:
                run_index = int(run_index) - 1
                if 0 <= run_index < len(runs):
                    args.run_id = runs[run_index].info.run_id
                else:
                    print("Invalid run number.")
                    exit(1)
            except ValueError:
                print("Invalid input.")
                exit(1)
    
    if args.run_id:
        # Load model from MLflow
        model, run = load_model_from_mlflow(args.run_id, args.model_path)
        
        # Print run information
        run_name = run.data.tags.get("mlflow.runName", "Unnamed run")
        print(f"\nLoaded model from run: {run_name}")
        
        # Evaluate model
        evaluate_model(model, args.num_samples)
    elif not args.list_runs:
        print("Please provide a run ID or use --list-runs to see available runs.")
        exit(1) 