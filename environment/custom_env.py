import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.env_checker import check_env


class StudySchedulerEnv(gym.Env):
    """A custom environment for study scheduling. The agent suggests study times to a student based on their preferences and consistency."""

    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(StudySchedulerEnv, self).__init__()
        # Define action and observation space
        
        self.action_space = spaces.Discrete(5)  # 5 discrete actions ie suggest morning, suggest afternoon, suggest evening, suggest night, no suggestion

        # Define observation space: 7 features
        # [days_since_study, day_of_week, acceptance_rate, 
        #  morning_pref, afternoon_pref, evening_pref, night_pref]
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0, 0, 0, 0]), high=np.array([7, 6, 1, 1, 1, 1, 1]), dtype=np.float32) 
                                            
        self.time_slots = ['morning', 'afternoon', 'evening', 'night']                                    

        self.current_day =  0 # Day of the week (0-6)
        self.days_since_study = 0 # Days since last study session

        """Model Student profile parameters"""
        self.student_preferred_time = 0 # Preferred time slot (0-3) corresponding to morning, afternoon, evening, night
        self.student_consistency = 0.7 # Probability of completing a study session once accepted
        self.acceptance_history = [] # History of accepted suggestions
        self.study_history = [] # History of study sessions

        self.consecutive_suggestions = 0 # Count of consecutive suggestions made
        self.time_preferences = [0, 0, 0, 0] # Preferences for each time slot

    # def reset(self):
    #     """Reset the state of the environment to an initial state."""
    #     self.state = np.random.rand(3)  # Random initial state
    #     self.steps_beyond_done = None
    #     return self.state, {}


    def reset(self, seed=None, options=None):
        """Reset environment to start a new episode (new week)."""
        super().reset(seed=seed) 
        
        # Generate new student profile
        self.student_preferred_time = self.np_random.integers(0, 4)  # Random preferred time
        self.student_consistency = self.np_random.uniform(0.5, 0.9)  #  Random Consistency level
        
        # Reset episode state
        self.current_day = 0
        self.days_since_study = 0
        self.acceptance_history = []
        self.study_history = []
        self.consecutive_suggestions = 0
        self.time_preferences = [0, 0, 0, 0]
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    

    # def step(self, action):
    #     """Execute one time step within the environment."""
    #     assert self.action_space.contains(action), "Invalid Action"

    #     Example dynamics: random next state
    #     self.state = np.random.rand(3)

    #     reward = 1.0 if action == 1 else 0.0  # Example reward structure
    #     done = np.random.rand() > 0.95  # Randomly end the episode

    #     return self.state, reward, done, False, {}

    def step(self, action):
        """Execute one time step (one day) within the environment."""
        reward = 0
        studied_today = False
        accepted = False
        
        # Action 4 is "no suggestion"
        if action == 4:
            # Small chance student studies on their own
            if self.np_random.random() < 0.15:
                studied_today = True
                reward = 2  # Small positive for self-motivation
            else:
                reward = -3  # Moderate negative for passivity
            self.consecutive_suggestions = 0
        else:
            # Agent made a suggestion (action 0-3 corresponds to time slots)
            suggested_time = action
            self.consecutive_suggestions += 1
            
            # Calculate acceptance probability
            acceptance_prob = self._calculate_acceptance_probability(suggested_time)
            
            # Student decides whether to accept
            if self.np_random.random() < acceptance_prob:
                accepted = True
                self.acceptance_history.append(1)
                
                # Student decides whether to actually complete the session
                completion_prob = self.student_consistency
                if self.np_random.random() < completion_prob:
                    studied_today = True
                    reward = 15  # Strong positive for successful study session
                    # Update time preferences based on completed session 
                    self.time_preferences = [0, 0, 0, 0]
                    self.time_preferences[suggested_time] = 1
                    
                    # Extra penalty reduction if matched preference
                    if suggested_time == self.student_preferred_time:
                        reward += 3
                else:
                    reward = 2  # Small reward for acceptance even if skipped
            else:
                # Student declined
                self.acceptance_history.append(0)
                reward = -2  # Stronger negative for failed suggestion
                
                # Extra penalty for not matching preferences
                if suggested_time != self.student_preferred_time:
                    reward -= 1
        
        # Penalty for over-suggesting (more than 2 consecutive)
        if self.consecutive_suggestions > 3:
            reward -= 1
        
        # Update days since last study
        if studied_today:
            self.study_history.append({
                'day': self.current_day,
                'time_slot': action if action < 4 else None
            })
            self.days_since_study = 0
        else:
            self.days_since_study += 1
        
        # Move to next day
        self.current_day += 1
        
        # Check if episode is done (7 days completed)
        terminated = self.current_day >= 7
        truncated = False
        
        observation = self._get_observation()
        info = self._get_info()
        info['accepted'] = accepted
        info['studied'] = studied_today
        
        return observation, reward, terminated, truncated, info
    

    def _calculate_acceptance_probability(self, suggested_time):
        """Calculate probability that student accepts the suggestion."""
        base_prob = 0.7 if suggested_time == self.student_preferred_time else 0.2 # Base acceptance probability 70% if the suggestiion matches the student's preference, else 20%
        
        # higher probability the student will study if they haven't in over 3 days
        if self.days_since_study > 3:
            base_prob += 0.2
        
        # more than 3 suggestions in a row reduces acceptance probability
        if self.consecutive_suggestions >= 3:
            base_prob -= 0.2
        
        # Ensure probability is in valid range
        return np.clip(base_prob, 0.0, 1.0)
    
    def _get_observation(self):
        """Construct the observation array."""
        # Calculate acceptance rate over recent history
        if len(self.acceptance_history) > 0:
            acceptance_rate = np.mean(self.acceptance_history[-7:])  # Mean acceptance over the Last week
        else:
            acceptance_rate = 0.5  # Neutral starting point
        
        observation = np.array([
            self.days_since_study,
            self.current_day,
            acceptance_rate,
            self.time_preferences[0],  # Morning preference flag
            self.time_preferences[1],  # Afternoon preference flag
            self.time_preferences[2],  # Evening preference flag
            self.time_preferences[3],  # Night preference flag
        ], dtype=np.float32)
        
        return observation
    
    def _get_info(self):
        """Return additional information (for debugging/logging)."""
        return {
            'current_day': self.current_day,
            'days_since_study': self.days_since_study,
            'student_preferred_time': self.time_slots[self.student_preferred_time],
            'consecutive_suggestions': self.consecutive_suggestions,
            'total_study_sessions': len(self.study_history)
        }
    
    def render(self, mode='human'):
        """Optional: Print current state (for debugging)."""
        if mode == 'human':
            print(f"Day {self.current_day}/7 | Days since study: {self.days_since_study} | "
                  f"Student prefers: {self.time_slots[self.student_preferred_time]}")


# Test the environment
if __name__ == "__main__":
    # Create environment
    env = StudySchedulerEnv()

    # Test with random actions - 10 experiments
    print("Testing environment to see reward structure...\n")
    episode_rewards = []
    
    for exp in range(10):
        obs, info = env.reset()
        
        total_reward = 0
        for day in range(7):
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        
        episode_rewards.append(total_reward)
        print(f"Experiment {exp+1}: Total reward: {total_reward:6.1f}, Study sessions: {info['total_study_sessions']}/7")

    print("\n" + "="*70)
    print("REWARD STRUCTURE ANALYSIS (10 Experiments)")
    print("="*70)
    print(f"Episode Rewards:        {episode_rewards}")
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    print(f"Average Reward:         {avg_reward:7.2f}")
    print(f"Min Reward:             {min(episode_rewards):7.2f}")
    print(f"Max Reward:             {max(episode_rewards):7.2f}")
    std_dev = (sum((x - avg_reward)**2 for x in episode_rewards) / len(episode_rewards))**0.5
    print(f"Std Dev:                {std_dev:7.2f}")
    print(f"Positive Episodes:      {sum(1 for r in episode_rewards if r > 0)}/10")
    print("="*70)
    
    check_env(env)  # Validate the custom environment
    print("Environment is valid!")


