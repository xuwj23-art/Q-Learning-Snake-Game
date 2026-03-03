"""
性能对比测试（无需GUI）
Performance Comparison without GUI
"""

import numpy as np
from snake_game import SnakeGameEnv
from q_learning_agent import QLearningAgent

def compare_performance():
    """对比随机策略和训练后的智能体"""
    
    # 创建环境
    env = SnakeGameEnv(width=10, height=10)
    
    print("=" * 60)
    print("性能评估：随机策略 vs Q-Learning智能体")
    print("=" * 60)
    
    # 测试随机策略
    print("\n测试随机策略（100局）...")
    random_scores = []
    
    for i in range(100):
        state = env.reset()
        done = False
        while not done:
            action_idx = np.random.randint(0, 3)
            action = np.zeros(3)
            action[action_idx] = 1
            state, reward, done, score = env.step(action)
        random_scores.append(score)
        
        if (i+1) % 20 == 0:
            print(f"  进度: {i+1}/100")
    
    print(f"\n随机策略结果:")
    print(f"  平均分: {np.mean(random_scores):.2f} ± {np.std(random_scores):.2f}")
    print(f"  最高分: {max(random_scores)}")
    print(f"  中位数: {np.median(random_scores):.0f}")
    
    # 测试Q-Learning智能体
    print("\n测试Q-Learning智能体（100局）...")
    agent = QLearningAgent()
    
    # 加载训练好的模型
    model_path = '../results/q_learning_snake_best.pkl'
    if not agent.load(model_path):
        print("错误：未找到训练好的模型！")
        return
    
    agent_scores = []
    
    for i in range(100):
        state = env.reset()
        done = False
        while not done:
            action = agent.get_action(state, training=False)
            state, reward, done, score = env.step(action)
        agent_scores.append(score)
        
        if (i+1) % 20 == 0:
            print(f"  进度: {i+1}/100")
    
    print(f"\nQ-Learning智能体结果:")
    print(f"  平均分: {np.mean(agent_scores):.2f} ± {np.std(agent_scores):.2f}")
    print(f"  最高分: {max(agent_scores)}")
    print(f"  中位数: {np.median(agent_scores):.0f}")
    
    # 计算性能提升
    improvement = ((np.mean(agent_scores) - np.mean(random_scores)) / np.mean(random_scores) * 100)
    
    print("\n" + "=" * 60)
    print(f"✨ 性能提升: {improvement:.1f}%")
    print("=" * 60)
    
    # 保存结果
    with open('../results/performance_comparison.txt', 'w', encoding='utf-8') as f:
        f.write("性能对比结果\n")
        f.write("=" * 60 + "\n\n")
        f.write("随机策略:\n")
        f.write(f"  平均分: {np.mean(random_scores):.2f} ± {np.std(random_scores):.2f}\n")
        f.write(f"  最高分: {max(random_scores)}\n")
        f.write(f"  中位数: {np.median(random_scores):.0f}\n\n")
        f.write("Q-Learning智能体:\n")
        f.write(f"  平均分: {np.mean(agent_scores):.2f} ± {np.std(agent_scores):.2f}\n")
        f.write(f"  最高分: {max(agent_scores)}\n")
        f.write(f"  中位数: {np.median(agent_scores):.0f}\n\n")
        f.write(f"性能提升: {improvement:.1f}%\n")
    
    print(f"\n结果已保存到: ../results/performance_comparison.txt")

if __name__ == "__main__":
    compare_performance()
