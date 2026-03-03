"""
Training script for Q-Learning Snake
"""

import numpy as np
import matplotlib.pyplot as plt
from snake_game import SnakeGameEnv
from q_learning_agent import QLearningAgent
import time
import os

class Trainer:
    
    def __init__(self, env, agent, episodes=1000, save_interval=100):
        self.env = env
        self.agent = agent
        self.episodes = episodes
        self.save_interval = save_interval
        
        # track training progress
        self.scores = []
        self.avg_scores = []
        self.epsilons = []
        self.q_table_sizes = []
        
    def train(self, save_path='../results/q_learning_snake.pkl'):
        print("=" * 60)
        print("Training Q-Learning Snake Agent")
        print("=" * 60)
        print(f"Episodes: {self.episodes}")
        print(f"Learning rate: {self.agent.learning_rate}")
        print(f"Discount factor: {self.agent.discount_factor}")
        print(f"Initial epsilon: {self.agent.epsilon}")
        print(f"Epsilon decay: {self.agent.epsilon_decay}")
        print("=" * 60)
        
        start_time = time.time()
        best_score = 0
        
        for episode in range(1, self.episodes + 1):
            state = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done:
                # pick action
                action = self.agent.get_action(state, training=True)
                
                # take action
                next_state, reward, done, score = self.env.step(action)
                
                # update Q-table
                self.agent.update(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                steps += 1
            
            # decay epsilon after each episode
            self.agent.decay_epsilon()
            
            # record stats
            self.scores.append(score)
            self.epsilons.append(self.agent.epsilon)
            self.q_table_sizes.append(self.agent.get_q_table_size())
            
            # moving average
            avg_score = np.mean(self.scores[-100:]) if len(self.scores) >= 100 else np.mean(self.scores)
            self.avg_scores.append(avg_score)
            
            # save best model
            if score > best_score:
                best_score = score
                best_model_path = save_path.replace('.pkl', '_best.pkl')
                self.agent.save(best_model_path)
            
            # print progress
            if episode % 10 == 0 or episode == 1:
                elapsed_time = time.time() - start_time
                print(f"Episode: {episode:4d}/{self.episodes} | "
                      f"Score: {score:3d} | "
                      f"Avg: {avg_score:6.2f} | "
                      f"Best: {best_score:3d} | "
                      f"Epsilon: {self.agent.epsilon:.3f} | "
                      f"Q-Table: {self.agent.get_q_table_size():5d} | "
                      f"Time: {elapsed_time:.1f}s")
            
            # save checkpoint
            if episode % self.save_interval == 0:
                self.agent.save(save_path)
        
        # save final model
        self.agent.save(save_path)
        
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Total time: {total_time:.2f}s")
        print(f"Best score: {best_score}")
        print(f"Final avg: {avg_score:.2f}")
        print(f"Q-table size: {self.agent.get_q_table_size()}")
        print(f"Final epsilon: {self.agent.epsilon:.4f}")
        print("=" * 60)
        
        # plot results
        self.plot_training_results()
    
    def plot_training_results(self, save_dir='../results'):
        os.makedirs(save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # score curve
        axes[0, 0].plot(self.scores, alpha=0.3, label='Score')
        axes[0, 0].plot(self.avg_scores, label='Avg (100 eps)', linewidth=2)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Training Score Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # epsilon decay
        axes[0, 1].plot(self.epsilons, color='orange')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Epsilon')
        axes[0, 1].set_title('Epsilon Decay')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-table growth
        axes[1, 0].plot(self.q_table_sizes, color='green')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Q-Table Size')
        axes[1, 0].set_title('Q-Table Growth')
        axes[1, 0].grid(True, alpha=0.3)
        
        # score distribution
        recent_scores = self.scores[-100:] if len(self.scores) >= 100 else self.scores
        axes[1, 1].hist(recent_scores, bins=20, color='purple', alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Score Distribution (Last 100 Eps)')
        axes[1, 1].axvline(np.mean(recent_scores), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(recent_scores):.2f}')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, 'training_results.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nTraining plot saved to: {save_path}")
        
        plt.show()
    
    def get_training_stats(self):
        return {
            'total_episodes': len(self.scores),
            'max_score': max(self.scores) if self.scores else 0,
            'avg_score': np.mean(self.scores) if self.scores else 0,
            'final_epsilon': self.epsilons[-1] if self.epsilons else 0,
            'q_table_size': self.q_table_sizes[-1] if self.q_table_sizes else 0
        }


def main():
    # setup
    env = SnakeGameEnv(width=10, height=10)
    
    agent = QLearningAgent(
        state_size=11,
        action_size=3,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    trainer = Trainer(
        env=env,
        agent=agent,
        episodes=500,
        save_interval=50
    )
    
    # train
    trainer.train(save_path='../results/q_learning_snake.pkl')
    
    # print final stats
    print("\nFinal Stats:")
    stats = trainer.get_training_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
