
import numpy as np
import pickle
import os
from collections import defaultdict

class QLearningAgent:
    """Q-Learning强化学习智能体"""
    
    def __init__(self, state_size=11, action_size=3, learning_rate=0.1, 
                 discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995, 
                 epsilon_min=0.01):
        """
        初始化Q-Learning智能体
        
        Args:
            state_size: 状态空间大小
            action_size: 动作空间大小
            learning_rate: 学习率 (α)
            discount_factor: 折扣因子 (γ)
            epsilon: 初始探索率
            epsilon_decay: 探索率衰减
            epsilon_min: 最小探索率
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q表：使用字典存储 state -> action values
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        
        # 统计信息
        self.training_step = 0
        
    def get_action(self, state, training=True):
        """
        根据当前状态选择动作（Epsilon-Greedy策略）
        
        Args:
            state: 当前状态
            training: 是否在训练模式
            
        Returns:
            action: 选择的动作（one-hot编码）
        """
        state_key = self._state_to_key(state)
        
        # Epsilon-greedy策略
        if training and np.random.rand() <= self.epsilon:
            # 探索：随机选择动作
            action_idx = np.random.randint(self.action_size)
        else:
            # 利用：选择Q值最大的动作
            q_values = self.q_table[state_key]
            action_idx = np.argmax(q_values)
        
        # 转换为one-hot编码
        action = np.zeros(self.action_size, dtype=int)
        action[action_idx] = 1
        
        return action
    
    def update(self, state, action, reward, next_state, done):
        """
        更新Q表（Q-Learning更新规则）
        
        Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否终止
        """
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        action_idx = np.argmax(action)
        
        # 获取当前Q值
        current_q = self.q_table[state_key][action_idx]
        
        # 计算目标Q值
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[next_state_key])
            target_q = reward + self.discount_factor * max_next_q
        
        # 更新Q值
        self.q_table[state_key][action_idx] += self.learning_rate * (target_q - current_q)
        
        self.training_step += 1
    
    def decay_epsilon(self):
        """衰减探索率"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def _state_to_key(self, state):
        """将状态数组转换为可哈希的键"""
        return tuple(state)
    
    def save(self, filepath):
        """保存Q表和参数"""
        data = {
            'q_table': dict(self.q_table),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"模型已保存到: {filepath}")
    
    def load(self, filepath):
        """加载Q表和参数"""
        if not os.path.exists(filepath):
            print(f"文件不存在: {filepath}")
            return False
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.q_table = defaultdict(lambda: np.zeros(self.action_size), data['q_table'])
        self.epsilon = data['epsilon']
        self.training_step = data['training_step']
        self.learning_rate = data['learning_rate']
        self.discount_factor = data['discount_factor']
        
        print(f"模型已加载: {filepath}")
        return True
    
    def get_q_table_size(self):
        """返回Q表的大小（已探索的状态数量）"""
        return len(self.q_table)
    
    def get_stats(self):
        """获取智能体统计信息"""
        return {
            'q_table_size': self.get_q_table_size(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor
        }


if __name__ == "__main__":
    # 测试Q-Learning智能体
    print("测试Q-Learning智能体...")
    
    agent = QLearningAgent(state_size=11, action_size=3)
    
    # 创建模拟状态
    state = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1])
    next_state = np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1])
    
    print(f"\n初始统计信息:")
    print(agent.get_stats())
    
    # 测试动作选择
    action = agent.get_action(state, training=True)
    print(f"\n选择的动作: {action}")
    
    # 测试Q表更新
    reward = 1.0
    agent.update(state, action, reward, next_state, done=False)
    print(f"\n更新后Q表大小: {agent.get_q_table_size()}")
    
    # 测试保存和加载
    test_path = "../results/test_agent.pkl"
    agent.save(test_path)
    
    new_agent = QLearningAgent()
    new_agent.load(test_path)
    print(f"\n加载后统计信息:")
    print(new_agent.get_stats())
    
    # 清理测试文件
    if os.path.exists(test_path):
        os.remove(test_path)
    
    print("\nQ-Learning智能体测试完成！")
