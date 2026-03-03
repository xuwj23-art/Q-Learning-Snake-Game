"""
可视化工具 - 使用Pygame显示游戏
Visualization Tool with Pygame
"""

import pygame
import numpy as np
from snake_game import SnakeGameEnv, Direction
from q_learning_agent import QLearningAgent
import time

# 颜色定义
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (213, 50, 80)
GREEN = (0, 255, 0)
BLUE = (50, 153, 213)
YELLOW = (255, 255, 0)
DARK_GREEN = (0, 200, 0)
GRAY = (128, 128, 128)

class SnakeGameUI:
    """贪吃蛇游戏UI类"""
    
    def __init__(self, env, agent=None, block_size=30, speed=10):
        """
        初始化游戏UI
        
        Args:
            env: 游戏环境
            agent: Q-Learning智能体（可选）
            block_size: 每个格子的像素大小
            speed: 游戏速度（FPS）
        """
        self.env = env
        self.agent = agent
        self.block_size = block_size
        self.speed = speed
        
        # 计算窗口尺寸
        self.width = env.width * block_size
        self.height = env.height * block_size + 100  # 额外空间显示信息
        
        # 初始化Pygame
        pygame.init()
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Q-Learning 贪吃蛇')
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont('simhei', 25)
        self.font_small = pygame.font.SysFont('simhei', 18)
        
        # 游戏统计
        self.game_count = 0
        self.total_score = 0
        self.high_score = 0
        
    def play_step(self, action=None):
        """
        执行一步游戏
        
        Args:
            action: 动作（如果为None，则由智能体决定）
        """
        # 处理pygame事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None, None, True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return None, None, True
        
        # 获取动作
        if action is None and self.agent is not None:
            state = self.env._get_state()
            action = self.agent.get_action(state, training=False)
        elif action is None:
            action = [1, 0, 0]  # 默认直行
        
        # 执行动作
        state, reward, done, score = self.env.step(action)
        
        # 更新UI
        self._update_ui(score)
        self.clock.tick(self.speed)
        
        return state, reward, done
    
    def _update_ui(self, score):
        """更新UI显示"""
        self.display.fill(BLACK)
        
        # 绘制蛇
        for i, point in enumerate(self.env.snake):
            x = point.x * self.block_size
            y = point.y * self.block_size
            
            if i == 0:  # 蛇头
                pygame.draw.rect(self.display, DARK_GREEN, 
                               pygame.Rect(x, y, self.block_size, self.block_size))
                pygame.draw.rect(self.display, GREEN, 
                               pygame.Rect(x+4, y+4, self.block_size-8, self.block_size-8))
            else:  # 蛇身
                pygame.draw.rect(self.display, GREEN, 
                               pygame.Rect(x, y, self.block_size, self.block_size))
                pygame.draw.rect(self.display, DARK_GREEN, 
                               pygame.Rect(x+4, y+4, self.block_size-8, self.block_size-8))
        
        # 绘制食物
        food_x = self.env.food.x * self.block_size
        food_y = self.env.food.y * self.block_size
        pygame.draw.rect(self.display, RED, 
                       pygame.Rect(food_x, food_y, self.block_size, self.block_size))
        pygame.draw.circle(self.display, YELLOW, 
                         (food_x + self.block_size//2, food_y + self.block_size//2), 
                         self.block_size//3)
        
        # 绘制网格线（可选）
        for x in range(0, self.width, self.block_size):
            pygame.draw.line(self.display, GRAY, (x, 0), (x, self.env.height * self.block_size))
        for y in range(0, self.env.height * self.block_size, self.block_size):
            pygame.draw.line(self.display, GRAY, (0, y), (self.width, y))
        
        # 显示信息
        info_y = self.env.height * self.block_size + 10
        
        # 当前得分
        score_text = self.font_large.render(f'得分: {score}', True, WHITE)
        self.display.blit(score_text, [10, info_y])
        
        # 最高分
        high_score_text = self.font_large.render(f'最高分: {self.high_score}', True, YELLOW)
        self.display.blit(high_score_text, [10, info_y + 30])
        
        # 游戏次数
        game_count_text = self.font_small.render(f'游戏次数: {self.game_count}', True, WHITE)
        self.display.blit(game_count_text, [self.width - 150, info_y])
        
        # 智能体信息
        if self.agent is not None:
            epsilon_text = self.font_small.render(f'Epsilon: {self.agent.epsilon:.3f}', True, BLUE)
            self.display.blit(epsilon_text, [self.width - 150, info_y + 25])
            
            q_size_text = self.font_small.render(f'Q-Table: {self.agent.get_q_table_size()}', True, BLUE)
            self.display.blit(q_size_text, [self.width - 150, info_y + 50])
        
        pygame.display.flip()
    
    def play_game(self, num_games=10, auto=True):
        """
        玩多局游戏
        
        Args:
            num_games: 游戏局数
            auto: 是否自动模式（使用智能体）
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
                
                if state is None:  # 窗口被关闭
                    return
            
            # 更新统计
            self.game_count += 1
            self.total_score += self.env.score
            if self.env.score > self.high_score:
                self.high_score = self.env.score
            
            print(f"游戏 {self.game_count}: 得分 = {self.env.score}")
            
            # 游戏间短暂暂停
            time.sleep(0.5)
        
        # 显示最终统计
        print(f"\n游戏结束！")
        print(f"总游戏次数: {self.game_count}")
        print(f"最高分: {self.high_score}")
        print(f"平均分: {self.total_score / self.game_count:.2f}")
        
        # 等待关闭
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or \
                   (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    waiting = False
        
        pygame.quit()


def demo_random_play():
    """演示：随机玩游戏"""
    print("演示：随机策略")
    env = SnakeGameEnv(width=15, height=15)
    ui = SnakeGameUI(env, speed=10)
    ui.play_game(num_games=5, auto=False)


def demo_trained_agent():
    """演示：训练好的智能体"""
    print("演示：训练后的Q-Learning智能体")
    
    # 创建环境和智能体
    env = SnakeGameEnv(width=10, height=10)
    agent = QLearningAgent()
    
    # 加载训练好的模型
    model_path = '../results/q_learning_snake_best.pkl'
    if agent.load(model_path):
        ui = SnakeGameUI(env, agent, speed=15)
        ui.play_game(num_games=10, auto=True)
    else:
        print(f"未找到训练好的模型: {model_path}")
        print("请先运行 train.py 进行训练")


def compare_performance():
    """比较随机策略和训练后智能体的性能"""
    print("性能比较：随机策略 vs Q-Learning智能体")
    
    # 测试随机策略
    print("\n测试随机策略...")
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
    
    print(f"随机策略平均得分: {np.mean(random_scores):.2f} ± {np.std(random_scores):.2f}")
    print(f"随机策略最高分: {max(random_scores)}")
    
    # 测试训练后的智能体
    print("\n测试Q-Learning智能体...")
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
        
        print(f"Q-Learning平均得分: {np.mean(agent_scores):.2f} ± {np.std(agent_scores):.2f}")
        print(f"Q-Learning最高分: {max(agent_scores)}")
        
        improvement = (np.mean(agent_scores) - np.mean(random_scores)) / np.mean(random_scores) * 100
        print(f"\n性能提升: {improvement:.1f}%")
    else:
        print(f"未找到模型文件: {model_path}")


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
            print("用法: python visualize.py [random|agent|compare]")
    else:
        # 默认演示训练后的智能体
        demo_trained_agent()
