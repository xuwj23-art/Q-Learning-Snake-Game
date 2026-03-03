"""
Visualization tool with Pygame
"""

import pygame
import numpy as np
from snake_game import SnakeGameEnv, Direction
from q_learning_agent import QLearningAgent
import time

# colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (213, 50, 80)
GREEN = (0, 255, 0)
BLUE = (50, 153, 213)
YELLOW = (255, 255, 0)
DARK_GREEN = (0, 200, 0)
GRAY = (128, 128, 128)

class SnakeGameUI:
    
    def __init__(self, env, agent=None, block_size=30, speed=10):
        self.env = env
        self.agent = agent
        self.block_size = block_size
        self.speed = speed
        
        # window size
        self.width = env.width * block_size
        self.height = env.height * block_size + 100  # extra space for info
        
        # setup pygame
        pygame.init()
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Q-Learning Snake')
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont('arial', 25)
        self.font_small = pygame.font.SysFont('arial', 18)
        
        # game stats
        self.game_count = 0
        self.total_score = 0
        self.high_score = 0
        
    def play_step(self, action=None):
        """
        Play one step of the game
        """
        # handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None, None, True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return None, None, True
        
        # get action
        if action is None and self.agent is not None:
            state = self.env._get_state()
            action = self.agent.get_action(state, training=False)
        elif action is None:
            action = [1, 0, 0]  # default: go straight
        
        # take action
        state, reward, done, score = self.env.step(action)
        
        # update display
        self._update_ui(score)
        self.clock.tick(self.speed)
        
        return state, reward, done
    
    def _update_ui(self, score):
        self.display.fill(BLACK)
        
        # draw snake
        for i, point in enumerate(self.env.snake):
            x = point.x * self.block_size
            y = point.y * self.block_size
            
            if i == 0:  # head
                pygame.draw.rect(self.display, DARK_GREEN, 
                               pygame.Rect(x, y, self.block_size, self.block_size))
                pygame.draw.rect(self.display, GREEN, 
                               pygame.Rect(x+4, y+4, self.block_size-8, self.block_size-8))
            else:  # body
                pygame.draw.rect(self.display, GREEN, 
                               pygame.Rect(x, y, self.block_size, self.block_size))
                pygame.draw.rect(self.display, DARK_GREEN, 
                               pygame.Rect(x+4, y+4, self.block_size-8, self.block_size-8))
        
        # draw food
        food_x = self.env.food.x * self.block_size
        food_y = self.env.food.y * self.block_size
        pygame.draw.rect(self.display, RED, 
                       pygame.Rect(food_x, food_y, self.block_size, self.block_size))
        pygame.draw.circle(self.display, YELLOW, 
                         (food_x + self.block_size//2, food_y + self.block_size//2), 
                         self.block_size//3)
        
        # draw grid
        for x in range(0, self.width, self.block_size):
            pygame.draw.line(self.display, GRAY, (x, 0), (x, self.env.height * self.block_size))
        for y in range(0, self.env.height * self.block_size, self.block_size):
            pygame.draw.line(self.display, GRAY, (0, y), (self.width, y))
        
        # display info
        info_y = self.env.height * self.block_size + 10
        
        # score
        score_text = self.font_large.render(f'Score: {score}', True, WHITE)
        self.display.blit(score_text, [10, info_y])
        
        # high score
        high_score_text = self.font_large.render(f'Best: {self.high_score}', True, YELLOW)
        self.display.blit(high_score_text, [10, info_y + 30])
        
        # game count
        game_count_text = self.font_small.render(f'Games: {self.game_count}', True, WHITE)
        self.display.blit(game_count_text, [self.width - 150, info_y])
        
        # agent info
        if self.agent is not None:
            epsilon_text = self.font_small.render(f'Epsilon: {self.agent.epsilon:.3f}', True, BLUE)
            self.display.blit(epsilon_text, [self.width - 150, info_y + 25])
            
            q_size_text = self.font_small.render(f'Q-Table: {self.agent.get_q_table_size()}', True, BLUE)
            self.display.blit(q_size_text, [self.width - 150, info_y + 50])
        
        pygame.display.flip()
    
    def play_game(self, num_games=10, auto=True):
        """
        Play multiple games
        """
        for game in range(num_games):
            state = self.env.reset()
            done = False
            
            while not done:
                if auto and self.agent is not None:
                    action = self.agent.get_action(state, training=False)
                else:
                    action = None
                
                state, reward, done = self.play_step(action)
                
                if state is None:  # window closed
                    return
            
            # update stats
            self.game_count += 1
            self.total_score += self.env.score
            if self.env.score > self.high_score:
                self.high_score = self.env.score
            
            print(f"Game {self.game_count}: Score = {self.env.score}")
            
            # short pause between games
            time.sleep(0.5)
        
        # show final stats
        print(f"\nDone!")
        print(f"Total games: {self.game_count}")
        print(f"Best score: {self.high_score}")
        print(f"Average: {self.total_score / self.game_count:.2f}")
        
        # wait for close
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or \
                   (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    waiting = False
        
        pygame.quit()


def demo_random_play():
    """Demo: random strategy"""
    print("Demo: Random strategy")
    env = SnakeGameEnv(width=15, height=15)
    ui = SnakeGameUI(env, speed=10)
    ui.play_game(num_games=5, auto=False)


def demo_trained_agent():
    """Demo: trained agent"""
    print("Demo: Trained Q-Learning agent")
    
    env = SnakeGameEnv(width=10, height=10)
    agent = QLearningAgent()
    
    # load trained model
    model_path = '../results/q_learning_snake_best.pkl'
    if agent.load(model_path):
        ui = SnakeGameUI(env, agent, speed=15)
        ui.play_game(num_games=10, auto=True)
    else:
        print(f"Model not found: {model_path}")
        print("Run train.py first to train the agent")


def compare_performance():
    """Compare random vs trained agent"""
    print("Performance comparison: Random vs Q-Learning")
    
    # test random strategy
    print("\nTesting random strategy...")
    env = SnakeGameEnv(width=10, height=10)
    random_scores = []
    
    for i in range(100):
        state = env.reset()
        done = False
        while not done:
            action = np.random.randint(0, 3)
            action_onehot = np.zeros(3)
            action_onehot[action] = 1
            state, reward, done, score = env.step(action_onehot)
        random_scores.append(score)
    
    print(f"Random avg score: {np.mean(random_scores):.2f} ± {np.std(random_scores):.2f}")
    print(f"Random best: {max(random_scores)}")
    
    # test trained agent
    print("\nTesting Q-Learning agent...")
    agent = QLearningAgent()
    model_path = '../results/q_learning_snake_best.pkl'
    
    if agent.load(model_path):
        agent_scores = []
        
        for i in range(100):
            state = env.reset()
            done = False
            while not done:
                action = agent.get_action(state, training=False)
                state, reward, done, score = env.step(action)
            agent_scores.append(score)
        
        print(f"Q-Learning avg score: {np.mean(agent_scores):.2f} ± {np.std(agent_scores):.2f}")
        print(f"Q-Learning best: {max(agent_scores)}")
        
        improvement = (np.mean(agent_scores) - np.mean(random_scores)) / np.mean(random_scores) * 100
        print(f"\nImprovement: {improvement:.1f}%")
    else:
        print(f"Model not found: {model_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == 'random':
            demo_random_play()
        elif mode == 'agent':
            demo_trained_agent()
        elif mode == 'compare':
            compare_performance()
        else:
            print("Usage: python visualize.py [random|agent|compare]")
    else:
        # default: show trained agent
        demo_trained_agent()
