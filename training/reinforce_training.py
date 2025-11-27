import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from environment.custom_env import StudySchedulerEnv
from stable_baselines3.common.monitor import Monitor
import numpy as np
import matplotlib.pyplot as plt
import json
from collections import deque
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Hyperparameter configurations
CONFIGS = [
    {"learning_rate": 1e-3, "gamma": 0.99},
    {"learning_rate": 5e-4, "gamma": 0.99},
    {"learning_rate": 1e-4, "gamma": 0.99},
    {"learning_rate": 1e-3, "gamma": 0.95},
    {"learning_rate": 5e-4, "gamma": 0.95},
    {"learning_rate": 1e-3, "gamma": 0.999},
    {"learning_rate": 3e-4, "gamma": 0.99},
    {"learning_rate": 7e-4, "gamma": 0.99},
    {"learning_rate": 5e-4, "gamma": 0.98},
    {"learning_rate": 1e-4, "gamma": 0.95},
]

EVAL_FREQ = 5000
SAVE_FREQ = 10000
TOTAL_TIMESTEPS_DEFAULT = 50000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def obs_to_tensor(obs):
    obs_arr = np.array(obs).ravel()
    return torch.tensor(obs_arr, dtype=torch.float32, device=device).unsqueeze(0)

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)

def make_policy(env):
    obs_sample, _ = env.reset()
    obs_dim = np.array(obs_sample).ravel().shape[0]
    action_dim = env.action_space.n
    return PolicyNetwork(obs_dim, action_dim).to(device)

def select_action(policy, obs):
    logits = policy(obs)
    dist = Categorical(logits=logits)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action.item(), log_prob

def compute_returns(rewards, gamma):
    returns = []
    R = 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

def evaluate_model(policy, env, n_episodes=10):
    policy.eval()
    rewards = []
    with torch.no_grad():
        for _ in range(n_episodes):
            obs, _ = env.reset()
            episode_reward = 0.0
            terminated = False
            while not terminated:
                obs_t = obs_to_tensor(obs)
                logits = policy(obs_t)
                action = torch.argmax(logits).item()
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
            rewards.append(episode_reward)
    policy.train()
    return float(np.mean(rewards))

def train_reinforce(config, run_number, total_timesteps=TOTAL_TIMESTEPS_DEFAULT):
    print(f"\n{'='*60}")
    print(f"Training REINFORCE Run {run_number + 1}/10")
    print(f"Config: {config}")
    print(f"{'='*60}\n")
    
    log_dir = f"./logs/reinforce/run_{run_number}"
    model_dir = f"./models/reinforce/run_{run_number}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    env = StudySchedulerEnv()
    env = Monitor(env, log_dir)
    eval_env = StudySchedulerEnv()
    eval_env = Monitor(eval_env)
    
    policy = make_policy(env)
    optimizer = optim.Adam(policy.parameters(), lr=config["learning_rate"])
    gamma = config["gamma"]
    
    total_timesteps_done = 0
    best_eval_reward = -float("inf")
    checkpoint_idx = 0
    episode_rewards_deque = deque(maxlen=100)
    
    start_time = time.time()
    
    while total_timesteps_done < total_timesteps:
        # Collect one episode
        ep_log_probs = []
        ep_rewards = []
        obs, _ = env.reset()
        terminated = False
        
        while not terminated:
            obs_t = obs_to_tensor(obs)
            action, log_prob = select_action(policy, obs_t)
            obs, reward, terminated, truncated, info = env.step(action)
            
            ep_log_probs.append(log_prob)
            ep_rewards.append(reward)
            total_timesteps_done += 1
            
            if total_timesteps_done >= total_timesteps:
                break
        
        episode_rewards_deque.append(sum(ep_rewards))
        
        # Compute returns and update
        returns = compute_returns(ep_rewards, gamma)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
        
        log_probs_t = torch.stack(ep_log_probs)
        loss = -(log_probs_t * returns_t).sum()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        optimizer.step()
        
        # Eval
        if total_timesteps_done % EVAL_FREQ < len(ep_rewards):
            eval_reward = evaluate_model(policy, eval_env, n_episodes=10)
            print(f"[T={total_timesteps_done}] Eval: {eval_reward:.2f} (best {best_eval_reward:.2f})")
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                torch.save(policy.state_dict(), f"{model_dir}/best_model.pt")
        
        # Checkpoint
        if total_timesteps_done // SAVE_FREQ > checkpoint_idx:
            checkpoint_idx = total_timesteps_done // SAVE_FREQ
            torch.save(policy.state_dict(), f"{model_dir}/checkpoint_{checkpoint_idx}.pt")
    
    torch.save(policy.state_dict(), f"{model_dir}/final_model.pt")
    final_mean_reward = evaluate_model(policy, eval_env, n_episodes=100)
    
    results = {
        "config": config,
        "mean_reward": float(final_mean_reward),
        "run_number": run_number
    }
    
    with open(f"{model_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nRun {run_number + 1} Complete - Mean Reward: {final_mean_reward:.2f}\n")
    return results

def plot_results(all_results):
    learning_rates = [r["config"]["learning_rate"] for r in all_results]
    mean_rewards = [r["mean_reward"] for r in all_results]
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(len(all_results)), mean_rewards)
    plt.xlabel("Run Number")
    plt.ylabel("Mean Reward")
    plt.title("REINFORCE Performance Across Runs")
    plt.xticks(range(len(all_results)), [f"Run {i+1}" for i in range(len(all_results))], rotation=45)
    
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
    plt.title("Learning Rate Impact")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("./models/reinforce/comparison.png")

if __name__ == "__main__":
    print("Starting REINFORCE Training with 10 Configurations")
    print("=" * 60)
    
    os.makedirs("./models/reinforce", exist_ok=True)
    os.makedirs("./logs/reinforce", exist_ok=True)
    
    all_results = []
    
    for i, config in enumerate(CONFIGS):
        try:
            results = train_reinforce(config, i, total_timesteps=TOTAL_TIMESTEPS_DEFAULT)
            all_results.append(results)
        except Exception as e:
            print(f"Error in run {i+1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    with open("./models/reinforce/all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    if all_results:
        plot_results(all_results)
    
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