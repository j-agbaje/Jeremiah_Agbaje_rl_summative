# **Reinforcement Learning Summative Assignment Report**

**Student Name:** Jeremiah

**Video Recording:** [Link to your Video, 3 minutes max, Camera On, Share the entire Screen]

**GitHub Repository:** https://github.com/j-agbaje/Jeremiah_Agbaje_rl_summative

---

## **1. Project Overview**

This project implements an intelligent study scheduling system using reinforcement learning to optimize student study habits. The agent acts as a personalized assistant that suggests optimal study times (morning, afternoon, evening, night) based on observed student behavior. Four RL algorithms (DQN, PPO, A2C, REINFORCE) were trained and compared over 50,000 timesteps with 10 hyperparameter configurations each. The environment simulates a 7-day scheduling period where the agent must balance making suggestions without overwhelming the student. PPO achieved the best performance (14.69 mean reward), successfully learning to identify student preferences and maximize completed study sessions.

---

## **2. Environment Description**

### **2.1 Agent(s)**

The agent represents a study scheduling assistant that makes daily suggestions about when to study. It observes student behavior patterns and learns to recommend appropriate time slots while avoiding over-suggesting, which reduces acceptance rates.

### **2.2 Action Space**

The environment uses a **discrete action space** with 5 mutually exclusive actions that are executed once per day:

- **Action 0:** Suggest a morning study session (6 AM - 12 PM) - The agent recommends that the student could study during morning hours
- **Action 1:** Suggest an afternoon study session (12 PM - 5 PM) - The agent recommends that the student could study in the early-to-mid afternoon
- **Action 2:** Suggest an evening study session (5 PM - 9 PM) - The agent recommends that the student could study after work or school in the evening
- **Action 3:** Suggest a night study session (9 PM - 12 AM) - The agent recommends that the student could study late at night before sleep
- **Action 4:** Make no suggestion - The agent remains silent, allowing the student to exercise autonomy and self-motivation

The discrete nature enables clear decision boundaries that are appropriate for scheduling problems. Each action represents a distinct time preference that the agent must learn to match with individual student habits.

### **2.3 Observation Space**

The agent receives a 7-dimensional continuous observation vector that is encoded as floating-point values:

1. **Days since last study (0-7):** This tracks how long it has been since the student last completed a session. Higher values indicate an urgency for intervention. The value resets to 0 after a successful study session.

2. **Current day of week (0-6):** This provides temporal context (0=Monday, 6=Sunday). It helps the agent learn weekly patterns like "students are less likely to study on weekends."

3. **Acceptance rate (0-1):** This is a running average of suggestion acceptance over the last 7 suggestions. It indicates the agent's credibility with the student. Low rates signal that the agent is over-suggesting or has poor timing.

4. **Morning preference flag (0/1):** This is a binary indicator that is set to 1 if the student completed a morning session previously, and 0 otherwise. It helps the agent detect morning preference patterns.

5. **Afternoon preference flag (0/1):** This is a binary indicator for afternoon session completion history.

6. **Evening preference flag (0/1):** This is a binary indicator for evening session completion history.

7. **Night preference flag (0/1):** This is a binary indicator for night session completion history.

**Partial Observability:** The agent cannot directly observe the student's true preferred time slot (which is a hidden state) or their consistency level. These characteristics must be inferred through interaction, which makes exploration critical. This design models real-world uncertainty where students may not explicitly state their preferences.

### **2.4 Reward Structure**

The reward function guides the agent to maximize study sessions while maintaining a positive relationship with the student:

**Positive Rewards:**
- **+15:** The student accepts the suggestion AND completes the study session - This is the primary objective and is heavily rewarded to prioritize actual learning outcomes over mere acceptance.
- **+3:** A bonus is awarded when the suggested time matches the student's hidden preference - This encourages the agent to learn individual preferences rather than making generic suggestions.
- **+2:** The student accepts the suggestion but skips the session - This represents partial success; acceptance builds trust even if the follow-through fails.
- **+2:** The student self-initiates study when no suggestion was made (Action 4) - This rewards the agent for fostering student autonomy and intrinsic motivation.

**Negative Rewards:**
- **-2:** The student declines the suggestion - This indicates poor timing or over-suggesting. It teaches the agent to be selective about when to intervene.
- **-1:** An additional penalty is applied for suggesting a non-preferred time slot - This encourages the agent to learn and respect individual preferences through observation.
- **-1:** A penalty is applied for making more than 3 consecutive suggestions - This prevents the agent from becoming annoying or overbearing, which would damage long-term acceptance rates.
- **-3:** No activity occurs when the agent made no suggestion (Action 4) - This is a stronger penalty than rejection to discourage excessive passivity. The agent must find the right balance between suggesting and staying silent.

**Mathematical formulation:**
```
R(s,a) = {
    15 + 3    if accepted ∧ completed ∧ matched_preference
    15        if accepted ∧ completed
    2         if accepted ∧ skipped
    -2 - 1    if declined ∧ wrong_time
    -2        if declined
    -3        if no_activity (a=4)
}
Additional penalty: -1 if consecutive_suggestions > 3
```

### **2.5 Environment Visualization**

The Pygame visualization displays a 7-day timeline with daily boxes that show completed sessions (marked with green checkmarks), agent suggestions (shown in orange text), and student responses (color-coded: green=success, yellow=partial, red=rejected). The display includes a running total reward and highlights the current day.

*(A 30-second video demonstration is included in the GitHub repository showing both the random agent baseline and the best PPO agent)*

---

## **3. System Analysis And Design**

### **3.1 Deep Q-Network (DQN)**

**Architecture:** The implementation uses an MlpPolicy with a 7-neuron input layer, 2 hidden layers (each with 64 neurons and ReLU activation), and a 5-neuron output layer for Q-values.

**Features:**
- An experience replay buffer with configurable capacity (10,000-50,000)
- A target network that is updated every 1,000 steps
- Epsilon-greedy exploration (ε decays from 1.0 to 0.05)
- Training starts after collecting 1,000 steps
- Batch sizes range from 32 to 128
- The Adam optimizer is used

No modifications were made to the standard DQN algorithm.

### **3.2 Policy Gradient Methods**

**PPO:**
- Uses a shared MLP backbone (2 layers with 64 neurons each) with separate actor-critic heads
- Implements a clipped surrogate objective (clip_range: 0.1-0.3)
- Uses Generalized Advantage Estimation (GAE) with λ=0.95
- Performs 10 epochs per batch, with n_steps ranging from 1024 to 4096
- The entropy coefficient ranges from 0.0 to 0.01

**A2C:**
- Uses a synchronous actor-critic with a shared architecture
- Implements N-step returns (5-20 steps)
- The value function coefficient ranges from 0.25 to 1.0
- The entropy coefficient ranges from 0.0 to 0.01

**REINFORCE:**
- A custom PyTorch implementation with 2 hidden layers (128 neurons each)
- Uses Monte Carlo episode returns with normalization
- Implements policy gradient updates with gradient clipping (max_norm=0.5)
- Updates are performed at the episode level (7-day episodes)

---

## **4. Implementation**

### **4.1 DQN**

| Run | Learning Rate | Gamma | Replay Buffer | Batch Size | Exploration Frac | Final ε | Mean Reward |
|-----|--------------|-------|---------------|------------|-----------------|---------|-------------|
| 1   | 1e-3 | 0.99 | 10000 | 64 | 0.1 | 0.05 | 6.25 |
| 2   | 5e-4 | 0.99 | 10000 | 64 | 0.1 | 0.05 | **12.98** |
| 3   | 1e-4 | 0.99 | 10000 | 64 | 0.1 | 0.05 | -4.10 |
| 4   | 1e-3 | 0.95 | 10000 | 64 | 0.1 | 0.05 | 7.82 |
| 5   | 1e-3 | 0.99 | 50000 | 64 | 0.2 | 0.10 | 7.15 |
| 6   | 1e-3 | 0.99 | 10000 | 32 | 0.15 | 0.05 | 0.56 |
| 7   | 1e-3 | 0.99 | 10000 | 128 | 0.1 | 0.02 | 11.02 |
| 8   | 5e-4 | 0.99 | 10000 | 64 | 0.3 | 0.10 | 6.30 |
| 9   | 5e-4 | 0.95 | 50000 | 32 | 0.15 | 0.05 | 4.86 |
| 10  | 1e-4 | 0.99 | 50000 | 128 | 0.05 | 0.01 | -1.58 |

**Best Configuration:** Run 2 (lr=5e-4, γ=0.99, buffer=10000, batch=64)

**Parameter Justification:** The learning rate of 5e-4 balances stable Q-value updates without overshooting. A gamma of 0.99 prioritizes long-term rewards over the 7-day episode. The standard buffer size (10,000) provides sufficient diversity without memory overhead.

### **4.2 REINFORCE**

| Run | Learning Rate | Gamma | Mean Reward |
|-----|--------------|-------|-------------|
| 1   | 1e-3 | 0.99 | -15.75 |
| 2   | 5e-4 | 0.99 | -15.20 |
| 3   | 1e-4 | 0.99 | -15.10 |
| 4   | 1e-3 | 0.95 | -2.13 |
| 5   | 5e-4 | 0.95 | 0.15 |
| 6   | 1e-3 | 0.999 | -16.30 |
| 7   | 3e-4 | 0.99 | -15.60 |
| 8   | 7e-4 | 0.99 | -15.85 |
| 9   | 5e-4 | 0.98 | 0.08 |
| 10  | 1e-4 | 0.95 | **3.87** |

**Best Configuration:** Run 10 (lr=1e-4, γ=0.95)

**Parameter Justification:** A very low learning rate (1e-4) is necessary to manage the high gradient variance. The lower gamma (0.95) reduces variance by discounting distant rewards more heavily.

### **4.3 A2C**

| Run | Learning Rate | Gamma | N-Steps | Ent Coef | VF Coef | Mean Reward |
|-----|--------------|-------|---------|----------|---------|-------------|
| 1   | 7e-4 | 0.99 | 5 | 0.0 | 0.5 | 2.79 |
| 2   | 1e-3 | 0.99 | 5 | 0.0 | 0.5 | 9.13 |
| 3   | 3e-4 | 0.99 | 5 | 0.0 | 0.5 | 10.28 |
| 4   | 7e-4 | 0.95 | 5 | 0.0 | 0.5 | 6.22 |
| 5   | 7e-4 | 0.99 | 10 | 0.01 | 0.5 | 7.18 |
| 6   | 7e-4 | 0.99 | 20 | 0.0 | 0.25 | 6.53 |
| 7   | 7e-4 | 0.99 | 5 | 0.0 | 1.0 | **10.32** |
| 8   | 5e-4 | 0.99 | 10 | 0.01 | 0.5 | 9.96 |
| 9   | 1e-3 | 0.95 | 20 | 0.005 | 0.25 | 8.05 |
| 10  | 3e-4 | 0.99 | 5 | 0.0 | 0.75 | 8.18 |

**Best Configuration:** Run 7 (lr=7e-4, γ=0.99, n_steps=5, vf_coef=1.0)

**Parameter Justification:** A value function coefficient of 1.0 significantly improves performance by weighting the value loss equally with the policy loss. Small n_steps (5) work well for the short 7-day episodes.

### **4.4 PPO**

| Run | Learning Rate | Gamma | N-Steps | Batch Size | Clip Range | Ent Coef | Mean Reward |
|-----|--------------|-------|---------|------------|------------|----------|-------------|
| 1   | 3e-4 | 0.99 | 2048 | 64 | 0.2 | 0.0 | 9.57 |
| 2   | 1e-3 | 0.99 | 2048 | 64 | 0.2 | 0.0 | **14.69** |
| 3   | 1e-4 | 0.99 | 2048 | 64 | 0.2 | 0.0 | 10.07 |
| 4   | 3e-4 | 0.95 | 2048 | 64 | 0.2 | 0.0 | 4.99 |
| 5   | 3e-4 | 0.99 | 1024 | 32 | 0.2 | 0.01 | 10.55 |
| 6   | 3e-4 | 0.99 | 4096 | 128 | 0.2 | 0.0 | 9.93 |
| 7   | 3e-4 | 0.99 | 2048 | 64 | 0.3 | 0.0 | 6.91 |
| 8   | 3e-4 | 0.99 | 2048 | 64 | 0.1 | 0.01 | 10.10 |
| 9   | 5e-4 | 0.95 | 1024 | 32 | 0.25 | 0.005 | 5.66 |
| 10  | 1e-4 | 0.99 | 4096 | 128 | 0.15 | 0.0 | 4.75 |

**Best Configuration:** Run 2 (lr=1e-3, γ=0.99, n_steps=2048, batch=64, clip=0.2)

**Parameter Justification:** A higher learning rate (1e-3) enables faster policy updates while the clip range (0.2) prevents destructive changes. Large n_steps (2048) collect substantial experience before updates, which improves sample efficiency.

---

## **5. Results Discussion**

### **5.1 Cumulative Rewards**

**DQN Performance (Image 1):**

![DQN Performance Across Runs](image1)

The left subplot shows that Run 2 achieved the best performance (12.98), while Runs 3 and 10 failed with negative rewards. The right subplot reveals that a learning rate of 5e-4 is optimal—rates that are too high (1e-3) or too low (1e-4) degrade performance. DQN shows high sensitivity to the learning rate selection.

**PPO Performance (Image 2):**

![PPO Performance Across Runs](image2)

PPO demonstrates superior and consistent performance, with Run 2 reaching 14.69. Seven out of ten runs exceeded a reward of 9.0. The learning rate impact plot shows that PPO is robust across the range from 1e-4 to 1e-3, with 1e-3 producing the best results. PPO's clipped objective enables aggressive updates safely.

**REINFORCE Performance (Image 3):**

![REINFORCE Performance Across Runs](image3)

REINFORCE shows catastrophic failure with 6 out of 10 runs producing large negative rewards (ranging from -15 to -16). Only Run 10 achieved marginal success (3.87). The learning rate plot displays extreme volatility with no clear pattern. The high variance prevents effective learning within the training budget.

**A2C Performance (Image 4):**

![A2C Performance Across Runs](image4)

A2C achieves moderate success with a best performance of 10.32 (Run 7). The performance is variable but better than REINFORCE. The value function coefficient (vf_coef=1.0) proved to be more critical than the learning rate for optimal performance.

**Training Stability:**
PPO exhibits the most stable training with consistent positive rewards. DQN shows moderate stability but has higher hyperparameter sensitivity. A2C displays increased variance. REINFORCE demonstrates severe instability with no convergence guarantees.

### **5.2 Episodes To Converge**

- **PPO:** Converged within 20,000-25,000 timesteps (approximately 3,000-4,000 episodes), which is the fastest convergence
- **DQN:** Required 30,000-35,000 timesteps (approximately 4,500-5,000 episodes), which is a moderate speed
- **A2C:** Converged around 35,000-40,000 timesteps (approximately 5,000-6,000 episodes), which is a slower convergence
- **REINFORCE:** Failed to converge within the 50,000 timestep budget

PPO's clipped updates and multiple epochs per batch provide superior sample efficiency, which enables faster convergence than the other methods.

### **5.3 Generalization**

Testing was performed on 100 unseen episodes with varied student profiles:

| Method | Training Reward | Test Reward | Performance Drop |
|--------|----------------|-------------|------------------|
| PPO    | 14.69          | 13.24       | 9.9% (excellent) |
| DQN    | 12.98          | 10.76       | 17.1% (moderate) |
| A2C    | 10.32          | 8.45        | 18.1% (moderate) |
| REINFORCE | 3.87        | 2.09        | 46.0% (poor)     |

**PPO** generalizes best with only a 9.9% drop, which indicates robust policy learning. **DQN and A2C** show moderate generalization with drops of 17-18%. **REINFORCE** fails to generalize with a 46% drop, which confirms that no meaningful policy was learned.

PPO successfully adapted to different preferences within 2-3 days. DQN was more conservative and missed opportunities. A2C struggled with low-consistency students. REINFORCE showed essentially random behavior.

---

## **6. Conclusion and Discussion**

**PPO performed best** (14.69 mean reward) due to its stable clipped updates, excellent sample efficiency, and robust generalization. **DQN achieved respectable results** (12.98) with good replay-based learning but showed higher hyperparameter sensitivity. **A2C showed moderate success** (10.32) with reasonable learning but higher variance. **REINFORCE fundamentally failed** (3.87) due to catastrophically high variance.

**Strengths by Method:**
- **PPO:** Has stable learning, fast convergence (3,000 episodes), excellent generalization (9.9% drop), and is robust across hyperparameters
- **DQN:** Has good sample efficiency through replay, clear Q-value interpretability, and works well with discrete actions
- **A2C:** Has computational efficiency, synchronous updates that provide stability, and reasonable performance with tuning
- **REINFORCE:** Has theoretical simplicity and unbiased gradients (but is impractical)

**Weaknesses and Improvements for Best PPO Model:**

1. **Over-suggestion bias:** The model occasionally suggests for 3-4 consecutive days. *Fix:* Increase the penalty to -3 for making more than 3 consecutive suggestions.

2. **Slow initial exploration:** The model takes 1-2 days to identify preferences. *Fix:* Add curiosity-driven exploration or systematic time-slot exploration in the first 2 days.

3. **Limited long-term planning:** The 7-day episodes don't capture multi-week patterns. *Fix:* Extend the episodes to 14-21 days.

4. **Binary preference encoding:** The model doesn't capture preference strength. *Fix:* Use continuous preference scores (0-1) with recency weighting.

5. **No explanation capability:** The model cannot justify suggestions. *Fix:* Add an attention mechanism or train an auxiliary explanation network.

**Additional Improvements:**
- Extend training to 200,000 timesteps with learning rate decay
- Add intermediate rewards (+0.5) for maintaining good acceptance rates
- Implement curriculum learning: start with consistent students and gradually add variability
- Use a larger network (3×128 neurons) or an LSTM for temporal memory
- Add action masking to prevent making more than 2 consecutive suggestions (as a safety constraint)

PPO successfully learned effective scheduling policies that adapt to individual preferences, which makes it suitable for real-world deployment with the identified improvements.