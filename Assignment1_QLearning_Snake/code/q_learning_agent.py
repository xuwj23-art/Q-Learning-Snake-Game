import numpy as np
import pickle
import os
from collections import defaultdict

class QLearningAgent:
    
    def __init__(self, state_size=11, action_size=3, learning_rate=0.1, 
                 discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995, 
                 epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table stored as dict: state -> action values
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        
        self.training_step = 0
        
    def get_action(self, state, training=True):
        """
        Pick an action using epsilon-greedy
        """
        state_key = self._state_to_key(state)
        
        # epsilon-greedy: explore vs exploit
        if training and np.random.rand() <= self.epsilon:
            # explore - random action
            action_idx = np.random.randint(self.action_size)
        else:
            # exploit - pick best action
            q_values = self.q_table[state_key]
            action_idx = np.argmax(q_values)
        
        # convert to one-hot
        action = np.zeros(self.action_size, dtype=int)
        action[action_idx] = 1
        
        return action
    
    def update(self, state, action, reward, next_state, done):
        """
        Update Q-table using Q-learning formula:
        Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
        """
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        action_idx = np.argmax(action)
        
        current_q = self.q_table[state_key][action_idx]
        
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[next_state_key])
            target_q = reward + self.discount_factor * max_next_q
        
        # apply update
        self.q_table[state_key][action_idx] += self.learning_rate * (target_q - current_q)
        
        self.training_step += 1
    
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def _state_to_key(self, state):
        # make state hashable
        return tuple(state)
    
    def save(self, filepath):
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
        print(f"Model saved to: {filepath}")
    
    def load(self, filepath):
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return False
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.q_table = defaultdict(lambda: np.zeros(self.action_size), data['q_table'])
        self.epsilon = data['epsilon']
        self.training_step = data['training_step']
        self.learning_rate = data['learning_rate']
        self.discount_factor = data['discount_factor']
        
        print(f"Model loaded from: {filepath}")
        return True
    
    def get_q_table_size(self):
        return len(self.q_table)
    
    def get_stats(self):
        return {
            'q_table_size': self.get_q_table_size(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor
        }


if __name__ == "__main__":
    # quick test
    print("Testing Q-learning agent...")
    
    agent = QLearningAgent(state_size=11, action_size=3)
    
    # fake states for testing
    state = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1])
    next_state = np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1])
    
    print(f"\nInitial stats:")
    print(agent.get_stats())
    
    # test action selection
    action = agent.get_action(state, training=True)
    print(f"\nChosen action: {action}")
    
    # test Q-table update
    reward = 1.0
    agent.update(state, action, reward, next_state, done=False)
    print(f"\nQ-table size after update: {agent.get_q_table_size()}")
    
    # test save/load
    test_path = "../results/test_agent.pkl"
    agent.save(test_path)
    
    new_agent = QLearningAgent()
    new_agent.load(test_path)
    print(f"\nStats after loading:")
    print(new_agent.get_stats())
    
    # cleanup
    if os.path.exists(test_path):
        os.remove(test_path)
    
    print("\nAgent test complete!")
