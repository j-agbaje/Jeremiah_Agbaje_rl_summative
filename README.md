# Study Scheduler RL Agent

An intelligent study scheduling system that uses Reinforcement Learning to learn individual student behavior patterns and optimize study session suggestions.

## Project Overview

This project implements and compares four RL algorithms (DQN, PPO, A2C, REINFORCE) on a custom study scheduling environment. The agent learns to suggest optimal study times based on student preferences and behavior patterns.

## Environment Description

**Custom Gymnasium Environment: Study Scheduler**

- **Observation Space**: 7 features
  - Days since last study session (0-7)
  - Current day of week (0-6)
  - Student acceptance rate (0-1)
  - Time preference flags: morning, afternoon, evening, night (binary)

- **Action Space**: 5 discrete actions
  - 0-3: Suggest study time (morning/afternoon/evening/night)
  - 4: Make no suggestion

- **Rewards**:
  - +10: Student accepts and completes study session
  - +2: Student accepts but skips session
  - -2: Student declines suggestion
  - +1: Student studies independently
  - -1: No suggestion and no study

- **Episode**: 7 days (one week simulation)

## Installation

### Prerequisites
- Python 3.12
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/yourname_rl_summative.git
cd yourname_rl_summative
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Best Model Demo

To see the best-performing trained agent in action:

```bash
python main.py
```

This will:
- Load the best PPO model
- Run 5 demonstration episodes
- Display pygame visualization
- Show verbose terminal output with agent decisions and outcomes

### Training Models

To train individual algorithms:

```bash
# Train DQN
python training/dqn_training.py

# Train PPO
python training/ppo_training.py

# Train A2C
python training/a2c_training.py

# Train REINFORCE
python training/reinforce_training.py
```

Each training script:
- Runs 10 hyperparameter configurations
- Trains for 50,000 timesteps per configuration
- Saves models in `models/<algorithm>/`
- Saves logs in `logs/<algorithm>/`
- Generates comparison plots

### Testing the Environment

To visualize the environment with random actions:

```bash
python environment/rendering.py
```

## Results Summary

### Algorithm Performance

| Algorithm | Best Mean Reward | Configuration |
|-----------|------------------|---------------|
| **PPO** | **14.69** | lr=0.001, gamma=0.99, n_steps=2048 |
| DQN | 12.98 | lr=0.0005, gamma=0.99 |
| A2C | 10.32 | lr=0.0007, gamma=0.99, n_steps=5 |
| REINFORCE | 3.87 | lr=0.0001, gamma=0.95 |

**Winner**: PPO achieved the highest performance with consistent learning across episodes.

### Key Findings

1. **PPO performed best**: Stable learning with good exploration-exploitation balance
2. **DQN was competitive**: Strong performance but more variance
3. **A2C was reliable**: Consistent mid-range performance
4. **REINFORCE struggled**: High variance, required careful hyperparameter tuning

## Hyperparameters Tested

### DQN (10 configurations)
- Learning rates: 1e-4 to 1e-3
- Gamma: 0.95, 0.99
- Buffer sizes: 10k, 50k
- Batch sizes: 32, 64, 128
- Exploration strategies: varying epsilon schedules

### PPO (10 configurations)
- Learning rates: 1e-4 to 1e-3
- Gamma: 0.95, 0.99
- N-steps: 1024, 2048, 4096
- Clip range: 0.1 to 0.3
- Entropy coefficient: 0.0 to 0.01

### A2C (10 configurations)
- Learning rates: 3e-4 to 1e-3
- Gamma: 0.95, 0.99
- N-steps: 5, 10, 20
- Value coefficient: 0.25 to 1.0

### REINFORCE (10 configurations)
- Learning rates: 1e-4 to 1e-3
- Gamma: 0.95 to 0.999

## Technical Details

- **Framework**: Stable-Baselines3 (SB3) for DQN, PPO, A2C
- **REINFORCE**: Custom PyTorch implementation
- **Environment**: Custom Gymnasium environment
- **Visualization**: Pygame
- **Training**: Google Colab with A100 GPU
- **Total Training Time**: ~6-8 hours for all algorithms

## Capstone Connection

This RL exploration informs an adaptive study planner system that:
- Uses LSTM for temporal pattern prediction
- Incorporates conversational AI for user interaction
- Integrates with Google Calendar API
- Leverages insights from RL about student behavior modeling

The RL experiments validated the importance of:
- Learning individual preferences over time
- Balancing suggestion frequency to avoid fatigue
- Accounting for temporal patterns in behavior

## Dependencies

See `requirements.txt` for complete list. Key dependencies:
- gymnasium==1.0.0
- stable-baselines3==2.4.1
- torch==2.2.2
- numpy==1.26.4
- pygame==2.6.1

## Author

Jeremiah Agbaje 
j.agbaje@alustudent.com
African Leadership University

