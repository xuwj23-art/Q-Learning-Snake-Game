"""
Snake game environment for Q-learning
"""

import numpy as np
import random
from enum import Enum
from collections import namedtuple

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

class SnakeGameEnv:
    
    def __init__(self, width=20, height=20):
        self.width = width
        self.height = height
        self.reset()
        
    def reset(self):
        # start from center
        self.direction = Direction.RIGHT
        
        head_x = self.width // 2
        head_y = self.height // 2
        self.head = Point(head_x, head_y)
        
        # initial snake has 3 body parts
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
        # randomly place food, but not on snake
        while True:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            self.food = Point(x, y)
            if self.food not in self.snake:
                break
    
    def step(self, action):
        """
        Take an action and return new state, reward, done, score
        action: [1,0,0]=straight, [0,1,0]=right turn, [0,0,1]=left turn
        """
        self.frame_iteration += 1
        
        # update direction based on action
        self._update_direction(action)
        
        # move snake
        self._move()
        
        # check if game over
        reward = 0
        done = False
        
        # die if hit wall/self or takes too long
        if self._is_collision() or self.frame_iteration > 100 * len(self.snake):
            done = True
            reward = -10
            return self._get_state(), reward, done, self.score
        
        # check if ate food
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            # didn't eat, so remove tail
            self.snake.pop()
        
        # small penalty each step to encourage efficiency
        reward -= 0.01
        
        return self._get_state(), reward, done, self.score
    
    def _update_direction(self, action):
        # action: [1,0,0]=straight, [0,1,0]=right, [0,0,1]=left
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:  # left turn
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
        
        self.direction = new_dir
    
    def _move(self):
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
        if point is None:
            point = self.head
        
        # hit wall
        if point.x < 0 or point.x >= self.width or point.y < 0 or point.y >= self.height:
            return True
        
        # hit itself
        if point in self.snake[1:]:
            return True
        
        return False
    
    def _get_state(self):
        """
        Get current state as 11-dim vector
        - danger detection (front, right, left)
        - current direction (up, down, left, right)
        - food location (up, down, left, right)
        """
        head = self.head
        
        # points around head
        point_l = Point(head.x - 1, head.y)
        point_r = Point(head.x + 1, head.y)
        point_u = Point(head.x, head.y - 1)
        point_d = Point(head.x, head.y + 1)
        
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN
        
        state = [
            # danger straight ahead
            (dir_r and self._is_collision(point_r)) or 
            (dir_l and self._is_collision(point_l)) or 
            (dir_u and self._is_collision(point_u)) or 
            (dir_d and self._is_collision(point_d)),
            
            # danger on right
            (dir_u and self._is_collision(point_r)) or 
            (dir_d and self._is_collision(point_l)) or 
            (dir_l and self._is_collision(point_u)) or 
            (dir_r and self._is_collision(point_d)),
            
            # danger on left
            (dir_d and self._is_collision(point_r)) or 
            (dir_u and self._is_collision(point_l)) or 
            (dir_r and self._is_collision(point_u)) or 
            (dir_l and self._is_collision(point_d)),
            
            # current direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # food location relative to head
            self.food.x < head.x,  # food left
            self.food.x > head.x,  # food right
            self.food.y < head.y,  # food up
            self.food.y > head.y   # food down
        ]
        
        return np.array(state, dtype=int)
    
    def get_state_size(self):
        return 11
    
    def get_action_size(self):
        return 3


if __name__ == "__main__":
    # quick test
    print("Testing snake environment...")
    env = SnakeGameEnv(width=10, height=10)
    
    state = env.reset()
    print(f"Initial state: {state}")
    print(f"State size: {len(state)}")
    print(f"Snake position: {env.snake}")
    print(f"Food position: {env.food}")
    
    # try a few steps
    for i in range(5):
        action = [1, 0, 0]  # go straight
        state, reward, done, score = env.step(action)
        print(f"\nStep {i+1}:")
        print(f"  State: {state}")
        print(f"  Reward: {reward}")
        print(f"  Done: {done}")
        print(f"  Score: {score}")
        
        if done:
            print("Game over!")
            break
    
    print("\nTest complete!")
