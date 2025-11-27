import pygame
import sys
import os
import time
sys.path.insert(0, os.path.dirname(__file__))
from custom_env import StudySchedulerEnv

# Colors
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
BLUE = (50, 150, 255)
GREEN = (0, 200, 0)
RED = (255, 50, 50)
YELLOW = (255, 200, 0)
BLACK = (0, 0, 0)
ORANGE = (255, 165, 0)

def draw_visualization(screen, env, font, action_taken, accepted, studied, total_reward):
    """Draw the current state of the environment."""
    screen.fill(WHITE)
    
    # Title
    title = font.render("Study Scheduler - Random Agent Demo", True, BLACK)
    screen.blit(title, (20, 10))
    
    # Student preference (hidden info - for display only)
    pref_text = font.render(
        f"Student prefers: {env.time_slots[env.student_preferred_time]} (hidden from agent)",
        True, BLUE
    )
    screen.blit(pref_text, (20, 40))
    
    # Total reward
    reward_text = font.render(f"Total Reward: {total_reward:.1f}", True, BLACK)
    screen.blit(reward_text, (500, 40))
    
    # Draw 7-day timeline
    y_start = 100
    for day in range(7):
        x = 40 + day * 90
        
        # Day box
        color = GRAY
        if day == env.current_day:
            color = BLUE  # Current day highlighted
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
    
    # Show last action taken
    if action_taken is not None:
        action_y = y_start + 100
        if action_taken < 4:
            action_text = f"Suggested: {env.time_slots[action_taken]}"
            action_color = ORANGE
        else:
            action_text = "No suggestion"
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
        status = font.render("Episode Complete!", True, GREEN)
        screen.blit(status, (250, 220))
    
    pygame.display.flip()


if __name__ == "__main__":
    print("Starting Study Scheduler Visualization...")
    
    # Initialize pygame
    pygame.init()
    WIDTH, HEIGHT = 700, 300
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Study Scheduler Visualization")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)
    
    # Create environment
    env = StudySchedulerEnv()
    obs, info = env.reset()
    
    total_reward = 0
    action_taken = None
    accepted = False
    studied = False
    
    # Initial draw
    draw_visualization(screen, env, font, action_taken, accepted, studied, total_reward)
    time.sleep(2)  # Show initial state
    
    running = True
    while running and env.current_day < 7:
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
        
        if not running:
            break
        
        # Take random action
        action_taken = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action_taken)
        
        total_reward += reward
        accepted = info.get('accepted', False)
        studied = info.get('studied', False)
        
        print(f"Day {env.current_day}: Action={action_taken}, "
              f"Accepted={accepted}, Studied={studied}, Reward={reward:.1f}")
        
        # Draw updated state
        draw_visualization(screen, env, font, action_taken, accepted, studied, total_reward)
        
        # Wait between steps (1.5 seconds per day)
        time.sleep(5)
        clock.tick(60)  # 60 FPS for smooth rendering
    
    # Show final state
    if running:
        print(f"\nEpisode Complete! Total Reward: {total_reward:.1f}")
        print(f"Study sessions completed: {len(env.study_history)}/7 days")
        
        # Keep window open for 5 seconds
        for _ in range(50):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
            time.sleep(0.1)
    
    pygame.quit()
    print("Visualization closed.")



