"""
训练脚本
Training Script for Q-Learning Snake
"""

import numpy as np
import matplotlib.pyplot as plt
from snake_game import SnakeGameEnv
from q_learning_agent import QLearningAgent
import time
import os

class Trainer:
    """训练管理器"""
    
    def __init__(self, env, agent, episodes=1000, save_interval=100):
        """
        初始化训练器
        
        Args:
            env: 游戏环境
            agent: Q-Learning智能体
            episodes: 训练轮数
            save_interval: 保存模型的间隔
        """
        self.env = env
        self.agent = agent
        self.episodes = episodes
        self.save_interval = save_interval
        
        # 记录训练数据
        self.scores = []
        self.avg_scores = []
        self.epsilons = []
        self.q_table_sizes = []
        
    def train(self, save_path='../results/q_learning_snake.pkl'):
        """
        开始训练
        
        Args:
            save_path: 模型保存路径
        """
        print("=" * 60)
        print("开始训练Q-Learning贪吃蛇")
        print("=" * 60)
        print(f"训练轮数: {self.episodes}")
        print(f"学习率: {self.agent.learning_rate}")
        print(f"折扣因子: {self.agent.discount_factor}")
        print(f"初始Epsilon: {self.agent.epsilon}")
        print(f"Epsilon衰减率: {self.agent.epsilon_decay}")
        print("=" * 60)
        
        start_time = time.time()
        best_score = 0
        
        for episode in range(1, self.episodes + 1):
            state = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done:
                # 选择动作
                action = self.agent.get_action(state, training=True)
                
                # 执行动作
                next_state, reward, done, score = self.env.step(action)
                
                # 更新Q表
                self.agent.update(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                steps += 1
            
            # 衰减epsilon
            self.agent.decay_epsilon()
            
            # 记录数据
            self.scores.append(score)
            self.epsilons.append(self.agent.epsilon)
            self.q_table_sizes.append(self.agent.get_q_table_size())
            
            # 计算平均分
            avg_score = np.mean(self.scores[-100:]) if len(self.scores) >= 100 else np.mean(self.scores)
            self.avg_scores.append(avg_score)
            
            # 更新最佳分数
            if score > best_score:
                best_score = score
                # 保存最佳模型
                best_model_path = save_path.replace('.pkl', '_best.pkl')
                self.agent.save(best_model_path)
            
            # 打印进度
            if episode % 10 == 0 or episode == 1:
                elapsed_time = time.time() - start_time
                print(f"Episode: {episode:4d}/{self.episodes} | "
                      f"Score: {score:3d} | "
                      f"Avg Score: {avg_score:6.2f} | "
                      f"Best: {best_score:3d} | "
                      f"Epsilon: {self.agent.epsilon:.3f} | "
                      f"Q-Table: {self.agent.get_q_table_size():5d} | "
                      f"Time: {elapsed_time:.1f}s")
            
            # 定期保存模型
            if episode % self.save_interval == 0:
                self.agent.save(save_path)
        
        # 训练结束，保存最终模型
        self.agent.save(save_path)
        
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("训练完成！")
        print(f"总训练时间: {total_time:.2f}秒")
        print(f"最高分数: {best_score}")
        print(f"最终平均分: {avg_score:.2f}")
        print(f"最终Q表大小: {self.agent.get_q_table_size()}")
        print(f"最终Epsilon: {self.agent.epsilon:.4f}")
        print("=" * 60)
        
        # 绘制训练曲线
        self.plot_training_results()
    
    def plot_training_results(self, save_dir='../results'):
        """绘制训练结果"""
        os.makedirs(save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 得分曲线
        axes[0, 0].plot(self.scores, alpha=0.3, label='Score')
        axes[0, 0].plot(self.avg_scores, label='Average Score (100 episodes)', linewidth=2)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Training Score Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Epsilon衰减曲线
        axes[0, 1].plot(self.epsilons, color='orange')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Epsilon')
        axes[0, 1].set_title('Epsilon Decay Over Time')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Q表增长曲线
        axes[1, 0].plot(self.q_table_sizes, color='green')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Q-Table Size')
        axes[1, 0].set_title('Q-Table Growth Over Time')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 最近100轮的得分分布
        recent_scores = self.scores[-100:] if len(self.scores) >= 100 else self.scores
        axes[1, 1].hist(recent_scores, bins=20, color='purple', alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Score Distribution (Last 100 Episodes)')
        axes[1, 1].axvline(np.mean(recent_scores), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(recent_scores):.2f}')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        save_path = os.path.join(save_dir, 'training_results.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n训练曲线已保存到: {save_path}")
        
        plt.show()
    
    def get_training_stats(self):
        """获取训练统计信息"""
        return {
            'total_episodes': len(self.scores),
            'max_score': max(self.scores) if self.scores else 0,
            'avg_score': np.mean(self.scores) if self.scores else 0,
            'final_epsilon': self.epsilons[-1] if self.epsilons else 0,
            'q_table_size': self.q_table_sizes[-1] if self.q_table_sizes else 0
        }


def main():
    """主函数"""
    # 创建环境和智能体
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
    
    # 创建训练器
    trainer = Trainer(
        env=env,
        agent=agent,
        episodes=500,  # 可以根据需要调整
        save_interval=50
    )
    
    # 开始训练
    trainer.train(save_path='../results/q_learning_snake.pkl')
    
    # 打印最终统计
    print("\n最终训练统计:")
    stats = trainer.get_training_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
