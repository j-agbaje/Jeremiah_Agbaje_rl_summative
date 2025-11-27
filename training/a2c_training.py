import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from environment.custom_env import StudySchedulerEnv
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import numpy as np
import matplotlib.pyplot as plt
import json

# Hyperparameter configurations to test
CONFIGS = [
    {"learning_rate": 7e-4, "gamma": 0.99, "n_steps": 5, "ent_coef": 0.0, "vf_coef": 0.5},
    {"learning_rate": 1e-3, "gamma": 0.99, "n_steps": 5, "ent_coef": 0.0, "vf_coef": 0.5},
    {"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 5, "ent_coef": 0.0, "vf_coef": 0.5},
    {"learning_rate": 7e-4, "gamma": 0.95, "n_steps": 5, "ent_coef": 0.0, "vf_coef": 0.5},
    {"learning_rate": 7e-4, "gamma": 0.99, "n_steps": 10, "ent_coef": 0.01, "vf_coef": 0.5},
    {"learning_rate": 7e-4, "gamma": 0.99, "n_steps": 20, "ent_coef": 0.0, "vf_coef": 0.25},
    {"learning_rate": 7e-4, "gamma": 0.99, "n_steps": 5, "ent_coef": 0.0, "vf_coef": 1.0},
    {"learning_rate": 5e-4, "gamma": 0.99, "n_steps": 10, "ent_coef": 0.01, "vf_coef": 0.5},
    {"learning_rate": 1e-3, "gamma": 0.95, "n_steps": 20, "ent_coef": 0.005, "vf_coef": 0.25},
    {"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 5, "ent_coef": 0.0, "vf_coef": 0.75},
]

def train_a2c(config, run_number, total_timesteps=50000):
    """Train a single A2C model with given hyperparameters."""
    print(f"\n{'='*60}")
    print(f"Training A2C Run {run_number + 1}/10")
    print(f"Config: {config}")
    print(f"{'='*60}\n")
    
    # Create directories
    log_dir = f"./logs/a2c/run_{run_number}"
    model_dir = f"./models/a2c/run_{run_number}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Create environment
    env = StudySchedulerEnv()
    env = Monitor(env, log_dir)
    
    # Create eval environment
    eval_env = StudySchedulerEnv()
    eval_env = Monitor(eval_env)
    
    # Create model
    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        n_steps=config["n_steps"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        max_grad_norm=0.5,
        gae_lambda=0.95,
        verbose=1,
        tensorboard_log=log_dir
    )
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_dir,
        name_prefix="a2c_checkpoint"
    )
    
    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save final model
    model.save(f"{model_dir}/final_model")
    
    # Evaluate final performance
    mean_reward = evaluate_model(model, eval_env, n_episodes=100)
    
    # Save results
    results = {
        "config": config,
        "mean_reward": float(mean_reward),
        "run_number": run_number
    }
    
    with open(f"{model_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nRun {run_number + 1} Complete - Mean Reward: {mean_reward:.2f}\n")
    
    return results

def evaluate_model(model, env, n_episodes=100):
    """Evaluate trained model over n episodes."""
    rewards = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        terminated = False
        
        while not terminated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
        
        rewards.append(episode_reward)
    
    return np.mean(rewards)

def plot_results(all_results):
    """Plot comparison of all runs."""
    learning_rates = [r["config"]["learning_rate"] for r in all_results]
    mean_rewards = [r["mean_reward"] for r in all_results]
    
    plt.figure(figsize=(12, 6))
    
    # Plot 1: Mean rewards by run
    plt.subplot(1, 2, 1)
    plt.bar(range(len(all_results)), mean_rewards)
    plt.xlabel("Run Number")
    plt.ylabel("Mean Reward")
    plt.title("A2C Performance Across Runs")
    plt.xticks(range(len(all_results)), [f"Run {i+1}" for i in range(len(all_results))], rotation=45)
    
    # Plot 2: Learning rate vs performance
    plt.subplot(1, 2, 2)
    unique_lrs = sorted(set(learning_rates))
    lr_rewards = {lr: [] for lr in unique_lrs}
    for r in all_results:
        lr_rewards[r["config"]["learning_rate"]].append(r["mean_reward"])
    
    avg_rewards = [np.mean(lr_rewards[lr]) for lr in unique_lrs]
    plt.plot(unique_lrs, avg_rewards, 'o-')
    plt.xscale('log')
    plt.xlabel("Learning Rate")
    plt.ylabel("Average Mean Reward")
    plt.title("Learning Rate Impact on Performance")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("./models/a2c/comparison.png")
    print("Saved comparison plot to ./models/a2c/comparison.png")

if __name__ == "__main__":
    print("Starting A2C Training with 10 Hyperparameter Configurations")
    print("=" * 60)
    
    # Create base directories
    os.makedirs("./models/a2c", exist_ok=True)
    os.makedirs("./logs/a2c", exist_ok=True)
    
    all_results = []
    
    # Train with each configuration
    for i, config in enumerate(CONFIGS):
        try:
            results = train_a2c(config, i, total_timesteps=50000)
            all_results.append(results)
        except Exception as e:
            print(f"Error in run {i+1}: {e}")
            continue
    
    # Save all results
    with open("./models/a2c/all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Plot comparisons
    if all_results:
        plot_results(all_results)
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*60)
    for i, r in enumerate(all_results):
        print(f"Run {i+1}: Mean Reward = {r['mean_reward']:.2f}")
    
    if all_results:
        best_run = max(all_results, key=lambda x: x["mean_reward"])
        print(f"\nBest Run: {best_run['run_number'] + 1}")
        print(f"Best Config: {best_run['config']}")
        print(f"Best Mean Reward: {best_run['mean_reward']:.2f}")