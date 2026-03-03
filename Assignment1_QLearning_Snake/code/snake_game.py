"""
Snake Game Environment for Q-Learning
Simple grid-based game where snake eats food and avoids walls
"""

import numpy as np
import random
from enum import Enum
from collections import namedtuple

# Direction the snake can face
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Just a point with x and y coordinates
Point = namedtuple('Point', 'x, y')

class SnakeGameEnv:
    """
    The game environment where snake moves around
    """
    
    def __init__(self, width=20, height=20):
        """
        Set up the game board
        width and height are how many grid squares we have
        """
        self.width = width
        self.height = height
        self.reset()
        
    def reset(self):
        """重置游戏到初始状态"""
        # 初始化蛇的位置（从中心开始）
        self.direction = Direction.RIGHT
        
        head_x = self.width // 2
        head_y = self.height // 2
        self.head = Point(head_x, head_y)
        
        # 蛇的身体初始长度为3
        self.snake = [
            self.head,
            Point(self.head.x - 1, self.head.y),
            Point(self.head.x - 2, self.head.y)
        ]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        
        return self._get_state()
    
    def _place_food(self):
        """随机放置食物（不能在蛇身上）"""
        while True:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            self.food = Point(x, y)
            if self.food not in self.snake:
                break
    
    def step(self, action):
        """
        执行一个动作，返回下一个状态、奖励和是否结束
        
        Args:
            action: 动作 [1, 0, 0] = 直行, [0, 1, 0] = 右转, [0, 0, 1] = 左转
            
        Returns:
            state: 新的状态
            reward: 获得的奖励
            done: 游戏是否结束
            score: 当前得分
        """
        self.frame_iteration += 1
        
        # 1. 根据动作更新移动方向
        self._update_direction(action)
        
        # 2. 移动蛇
        self._move()
        
        # 3. 检查游戏是否结束
        reward = 0
        done = False
        
        if self._is_collision() or self.frame_iteration > 100 * len(self.snake):
            done = True
            reward = -10
            return self._get_state(), reward, done, self.score
        
        # 4. 检查是否吃到食物
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            # 移除蛇尾（没吃到食物就不增长）
            self.snake.pop()
        
        # 5. 计算距离食物的奖励（鼓励靠近食物）
        # reward += self._get_distance_reward()
        
        # 6. 小的负奖励，鼓励快速完成
        reward -= 0.01
        
        # 7. 返回新状态
        return self._get_state(), reward, done, self.score
    
    def _update_direction(self, action):
        """
        根据动作更新方向
        action: [1, 0, 0] = 直行, [0, 1, 0] = 右转, [0, 0, 1] = 左转
        """
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):  # 直行
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):  # 右转
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:  # [0, 0, 1] 左转
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
        
        self.direction = new_dir
    
    def _move(self):
        """移动蛇头"""
        x = self.head.x
        y = self.head.y
        
        if self.direction == Direction.RIGHT:
            x += 1
        elif self.direction == Direction.LEFT:
            x -= 1
        elif self.direction == Direction.DOWN:
            y += 1
        elif self.direction == Direction.UP:
            y -= 1
        
        self.head = Point(x, y)
        self.snake.insert(0, self.head)
    
    def _is_collision(self, point=None):
        """检查是否碰撞"""
        if point is None:
            point = self.head
        
        # 撞墙
        if point.x < 0 or point.x >= self.width or point.y < 0 or point.y >= self.height:
            return True
        
        # 撞自己
        if point in self.snake[1:]:
            return True
        
        return False
    
    def _get_state(self):
        """
        获取当前状态（11维状态向量）
        状态包括：
        - 危险检测（前、右、左）
        - 移动方向（上、下、左、右）
        - 食物位置（上、下、左、右）
        """
        head = self.head
        
        # 检查前方、右方、左方的危险
        point_l = Point(head.x - 1, head.y)
        point_r = Point(head.x + 1, head.y)
        point_u = Point(head.x, head.y - 1)
        point_d = Point(head.x, head.y + 1)
        
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN
        
        state = [
            # 前方危险
            (dir_r and self._is_collision(point_r)) or 
            (dir_l and self._is_collision(point_l)) or 
            (dir_u and self._is_collision(point_u)) or 
            (dir_d and self._is_collision(point_d)),
            
            # 右侧危险
            (dir_u and self._is_collision(point_r)) or 
            (dir_d and self._is_collision(point_l)) or 
            (dir_l and self._is_collision(point_u)) or 
            (dir_r and self._is_collision(point_d)),
            
            # 左侧危险
            (dir_d and self._is_collision(point_r)) or 
            (dir_u and self._is_collision(point_l)) or 
            (dir_r and self._is_collision(point_u)) or 
            (dir_l and self._is_collision(point_d)),
            
            # 当前移动方向
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # 食物位置
            self.food.x < head.x,  # 食物在左边
            self.food.x > head.x,  # 食物在右边
            self.food.y < head.y,  # 食物在上边
            self.food.y > head.y   # 食物在下边
        ]
        
        return np.array(state, dtype=int)
    
    def _get_distance_reward(self):
        """计算距离奖励（靠近食物给正奖励）"""
        # 计算曼哈顿距离
        current_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
        
        # 如果有上一个距离，比较变化
        if hasattr(self, 'prev_distance'):
            if current_distance < self.prev_distance:
                reward = 0.1  # 靠近食物
            else:
                reward = -0.1  # 远离食物
        else:
            reward = 0
        
        self.prev_distance = current_distance
        return reward
    
    def get_state_size(self):
        """返回状态空间大小"""
        return 11
    
    def get_action_size(self):
        """返回动作空间大小"""
        return 3


if __name__ == "__main__":
    # 测试游戏环境
    print("测试贪吃蛇游戏环境...")
    env = SnakeGameEnv(width=10, height=10)
    
    state = env.reset()
    print(f"初始状态: {state}")
    print(f"状态维度: {len(state)}")
    print(f"初始蛇位置: {env.snake}")
    print(f"食物位置: {env.food}")
    
    # 测试几步
    for i in range(5):
        action = [1, 0, 0]  # 直行
        state, reward, done, score = env.step(action)
        print(f"\n步骤 {i+1}:")
        print(f"  状态: {state}")
        print(f"  奖励: {reward}")
        print(f"  完成: {done}")
        print(f"  得分: {score}")
        
        if done:
            print("游戏结束！")
            break
    
    print("\n游戏环境测试完成！")
