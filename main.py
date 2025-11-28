import sys
import os
import pygame
import time
import numpy as np

from environment.custom_env import StudySchedulerEnv
from stable_baselines3 import PPO

# Colors
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
BLUE = (50, 150, 255)
GREEN = (0, 200, 0)
RED = (255, 50, 50)
YELLOW = (255, 200, 0)
BLACK = (0, 0, 0)
ORANGE = (255, 165, 0)

def draw_visualization(screen, env, font, action_taken, accepted, studied, total_reward, episode_num):
    """Draw the current state with detailed information."""
    screen.fill(WHITE)
    
    # Title and episode
    title = font.render(f"Study Scheduler - Best PPO Agent (Episode {episode_num})", True, BLACK)
    screen.blit(title, (20, 10))
    
    # Student preference and reward
    pref_text = font.render(
        f"Student prefers: {env.time_slots[env.student_preferred_time]}",
        True, BLUE
    )
    screen.blit(pref_text, (20, 40))
    
    reward_text = font.render(f"Total Reward: {total_reward:.1f}", True, BLACK)
    screen.blit(reward_text, (500, 40))
    
    # Draw 7-day timeline
    y_start = 100
    for day in range(7):
        x = 40 + day * 90
        
        # Day box
        color = GRAY
        if day == env.current_day:
            color = BLUE
        elif day < env.current_day:
            color = (220, 220, 220)
        pygame.draw.rect(screen, color, (x, y_start, 80, 80), 2)
        
        # Day label
        day_label = font.render(f"Day {day + 1}", True, BLACK)
        screen.blit(day_label, (x + 15, y_start - 25))
        
        # Check if student studied this day
        studied_this_day = any(sess['day'] == day for sess in env.study_history)
        if studied_this_day:
            pygame.draw.circle(screen, GREEN, (x + 40, y_start + 40), 25)
            check_font = pygame.font.SysFont(None, 40)
            check = check_font.render("✓", True, WHITE)
            screen.blit(check, (x + 25, y_start + 20))
    
    # Show action and result
    if action_taken is not None:
        action_y = y_start + 100
        if action_taken < 4:
            action_text = f"Agent suggested: {env.time_slots[action_taken]}"
            action_color = ORANGE
        else:
            action_text = "Agent: No suggestion"
            action_color = GRAY
        
        action_label = font.render(action_text, True, action_color)
        screen.blit(action_label, (40, action_y))
        
        # Show result
        if action_taken < 4:
            if accepted:
                if studied:
                    result = "✓ Accepted & Completed (+10)"
                    result_color = GREEN
                else:
                    result = "✓ Accepted but Skipped (+2)"
                    result_color = YELLOW
            else:
                result = "✗ Declined (-2)"
                result_color = RED
            
            result_label = font.render(result, True, result_color)
            screen.blit(result_label, (300, action_y))
    
    # Episode status
    if env.current_day >= 7:
        status = font.render(f"Episode Complete! Total: {total_reward:.1f}", True, GREEN)
        screen.blit(status, (200, 220))
    
    pygame.display.flip()

def run_best_model(model_path, n_episodes=5):
    """Run the best trained model with visualization."""
    print("\n" + "="*60)
    print("RUNNING BEST PERFORMING MODEL")
    print("="*60)
    print(f"Model: PPO Run 1")
    print(f"Configuration: lr=0.001, gamma=0.99, n_steps=2048")
    print(f"Mean Reward (training): 14.69")
    print("="*60 + "\n")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    print("Model loaded successfully!\n")
    
    # Initialize pygame
    pygame.init()
    WIDTH, HEIGHT = 700, 300
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Study Scheduler - Best PPO Agent")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)
    
    # Create environment
    env = StudySchedulerEnv()
    
    all_episode_rewards = []
    
    for episode in range(n_episodes):
        print(f"\n{'='*50}")
        print(f"Episode {episode + 1}/{n_episodes}")
        print('='*50)
        
        obs, info = env.reset()
        print(f"Student prefers: {env.time_slots[env.student_preferred_time]}")
        
        total_reward = 0
        action_taken = None
        accepted = False
        studied = False
        
        # Initial draw
        draw_visualization(screen, env, font, action_taken, accepted, studied, total_reward, episode + 1)
        time.sleep(2)
        
        running = True
        day = 0
        
        while running and day < 7:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    return
            
            # Predict action
            action, _states = model.predict(obs, deterministic=True)
            action_taken = int(action)
            
            # Print action
            if action_taken < 4:
                print(f"Day {day + 1}: Agent suggests {env.time_slots[action_taken]}")
            else:
                print(f"Day {day + 1}: Agent makes no suggestion")
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action_taken)
            total_reward += reward
            accepted = info.get('accepted', False)
            studied = info.get('studied', False)
            
            # Print result
            if action_taken < 4:
                if accepted:
                    if studied:
                        print(f"  → Student ACCEPTED and COMPLETED (reward: +10)")
                    else:
                        print(f"  → Student accepted but SKIPPED (reward: +2)")
                else:
                    print(f"  → Student DECLINED (reward: -2)")
            else:
                if studied:
                    print(f"  → Student studied on their own (reward: +1)")
                else:
                    print(f"  → No study session (reward: -1)")
            
            print(f"  Running total reward: {total_reward:.1f}")
            
            # Update visualization
            draw_visualization(screen, env, font, action_taken, accepted, studied, total_reward, episode + 1)
            
            day += 1
            time.sleep(2.5)  # Pause between days
            clock.tick(60)
        
        all_episode_rewards.append(total_reward)
        print(f"\nEpisode {episode + 1} Complete!")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Study Sessions Completed: {len(env.study_history)}/7 days")
        
        # Show final state for 3 seconds
        if running:
            for _ in range(30):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break
                time.sleep(0.1)
    
    # Final summary
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print(f"Episodes run: {n_episodes}")
    print(f"Episode rewards: {[f'{r:.2f}' for r in all_episode_rewards]}")
    print(f"Mean reward: {np.mean(all_episode_rewards):.2f}")
    print(f"Std reward: {np.std(all_episode_rewards):.2f}")
    print("="*60)
    
    print("\nPress any key to close...")
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                waiting = False
    
    pygame.quit()

if __name__ == "__main__":
    # Path to best model
    best_model_path = "./models/pg/ppo/run_4/best_model"
    
    # Check if model exists
    if not os.path.exists(best_model_path + ".zip"):
        print(f"ERROR: Model not found at {best_model_path}")
        print("Please ensure you have trained the PPO model first.")
        sys.exit(1)
    
    # Run demonstration
    run_best_model(best_model_path, n_episodes=5)