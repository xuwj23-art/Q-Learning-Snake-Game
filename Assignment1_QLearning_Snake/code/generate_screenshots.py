"""
使用Matplotlib生成游戏界面截图
Generate Game Screenshots using Matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from snake_game import SnakeGameEnv
from q_learning_agent import QLearningAgent

def draw_game_state(env, title="Snake Game", save_path=None):
    """
    绘制游戏状态
    
    Args:
        env: 游戏环境
        title: 图片标题
        save_path: 保存路径
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # 设置坐标轴
    ax.set_xlim(-0.5, env.width - 0.5)
    ax.set_ylim(-0.5, env.height - 0.5)
    ax.set_aspect('equal')
    
    # 绘制网格
    for i in range(env.width + 1):
        ax.axvline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
    for i in range(env.height + 1):
        ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
    
    # 绘制蛇身
    for i, point in enumerate(env.snake):
        if i == 0:  # 蛇头
            color = 'darkgreen'
            rect = patches.Rectangle((point.x - 0.4, point.y - 0.4), 0.8, 0.8,
                                     linewidth=2, edgecolor='black', facecolor=color)
        else:  # 蛇身
            color = 'lime'
            rect = patches.Rectangle((point.x - 0.4, point.y - 0.4), 0.8, 0.8,
                                     linewidth=1, edgecolor='darkgreen', facecolor=color)
        ax.add_patch(rect)
    
    # 绘制食物
    food_circle = patches.Circle((env.food.x, env.food.y), 0.3,
                                 linewidth=2, edgecolor='darkred', facecolor='red')
    ax.add_patch(food_circle)
    
    # 添加信息文本
    info_text = f"Score: {env.score} | Snake Length: {len(env.snake)}"
    ax.text(env.width / 2, -1.5, info_text, ha='center', fontsize=12, weight='bold')
    
    # 设置标题
    ax.set_title(title, fontsize=14, weight='bold', pad=20)
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    
    # 反转Y轴使得(0,0)在左上角
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"游戏画面已保存到: {save_path}")
    
    return fig


def generate_game_screenshots():
    """生成多个游戏状态的截图"""
    
    print("=" * 60)
    print("生成游戏界面截图")
    print("=" * 60)
    
    # 创建环境和智能体
    env = SnakeGameEnv(width=10, height=10)
    agent = QLearningAgent()
    
    # 加载训练好的模型
    model_path = '../results/q_learning_snake_best.pkl'
    if not agent.load(model_path):
        print("未找到训练模型，使用随机策略")
        agent = None
    
    # 截图1: 游戏初始状态
    print("\n生成截图1: 游戏初始状态")
    env.reset()
    draw_game_state(env, "Fig 2.1 - Initial Game State", 
                    '../results/screenshot_1_initial.png')
    plt.close()
    
    # 截图2: 游戏进行中（让智能体玩几步）
    print("生成截图2: 游戏进行中")
    env.reset()
    if agent:
        for _ in range(15):  # 玩15步
            state = env._get_state()
            action = agent.get_action(state, training=False)
            _, _, done, _ = env.step(action)
            if done:
                break
    
    draw_game_state(env, "Fig 2.2 - Game in Progress", 
                    '../results/screenshot_2_playing.png')
    plt.close()
    
    # 截图3: 游戏高分状态
    print("生成截图3: 游戏高分状态")
    # 尝试玩到较高分数
    best_score = 0
    best_env = None
    
    for attempt in range(10):  # 尝试10次，选择最好的
        temp_env = SnakeGameEnv(width=10, height=10)
        temp_env.reset()
        
        if agent:
            for _ in range(100):
                state = temp_env._get_state()
                action = agent.get_action(state, training=False)
                _, _, done, _ = temp_env.step(action)
                if done:
                    break
        
        if temp_env.score > best_score:
            best_score = temp_env.score
            best_env = temp_env
    
    if best_env:
        draw_game_state(best_env, f"Fig 2.3 - High Score Game (Score: {best_score})", 
                       '../results/screenshot_3_highscore.png')
        plt.close()
    
    print("\n" + "=" * 60)
    print("所有游戏截图已生成！")
    print("文件位置:")
    print("  - ../results/screenshot_1_initial.png")
    print("  - ../results/screenshot_2_playing.png")
    print("  - ../results/screenshot_3_highscore.png")
    print("=" * 60)


def generate_comparison_figure():
    """生成随机策略 vs 智能体的对比图"""
    
    print("\n生成性能对比图...")
    
    env = SnakeGameEnv(width=10, height=10)
    agent = QLearningAgent()
    
    # 加载模型
    model_path = '../results/q_learning_snake_best.pkl'
    agent.load(model_path)
    
    # 创建对比图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：随机策略
    env.reset()
    for _ in range(5):
        action_idx = np.random.randint(0, 3)
        action = np.zeros(3)
        action[action_idx] = 1
        _, _, done, _ = env.step(action)
        if done:
            break
    
    # 绘制随机策略
    ax = axes[0]
    ax.set_xlim(-0.5, env.width - 0.5)
    ax.set_ylim(-0.5, env.height - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    
    for i in range(env.width + 1):
        ax.axvline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
    for i in range(env.height + 1):
        ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
    
    for i, point in enumerate(env.snake):
        color = 'darkgreen' if i == 0 else 'lime'
        rect = patches.Rectangle((point.x - 0.4, point.y - 0.4), 0.8, 0.8,
                                 linewidth=1, edgecolor='black', facecolor=color)
        ax.add_patch(rect)
    
    food_circle = patches.Circle((env.food.x, env.food.y), 0.3,
                                 linewidth=2, edgecolor='darkred', facecolor='red')
    ax.add_patch(food_circle)
    
    ax.set_title(f'Random Strategy (Score: {env.score})', fontsize=12, weight='bold')
    ax.text(env.width / 2, -1, f"Snake Length: {len(env.snake)}", 
            ha='center', fontsize=10)
    
    # 右图：Q-Learning智能体
    env.reset()
    for _ in range(30):
        state = env._get_state()
        action = agent.get_action(state, training=False)
        _, _, done, _ = env.step(action)
        if done:
            break
    
    ax = axes[1]
    ax.set_xlim(-0.5, env.width - 0.5)
    ax.set_ylim(-0.5, env.height - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    
    for i in range(env.width + 1):
        ax.axvline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
    for i in range(env.height + 1):
        ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
    
    for i, point in enumerate(env.snake):
        color = 'darkgreen' if i == 0 else 'lime'
        rect = patches.Rectangle((point.x - 0.4, point.y - 0.4), 0.8, 0.8,
                                 linewidth=1, edgecolor='black', facecolor=color)
        ax.add_patch(rect)
    
    food_circle = patches.Circle((env.food.x, env.food.y), 0.3,
                                 linewidth=2, edgecolor='darkred', facecolor='red')
    ax.add_patch(food_circle)
    
    ax.set_title(f'Q-Learning Agent (Score: {env.score})', fontsize=12, weight='bold')
    ax.text(env.width / 2, -1, f"Snake Length: {len(env.snake)}", 
            ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('../results/screenshot_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("对比图已保存到: ../results/screenshot_comparison.png")


if __name__ == "__main__":
    # 生成游戏截图
    generate_game_screenshots()
    
    # 生成对比图
    generate_comparison_figure()
    
    print("\n✅ 所有游戏界面图片已生成完成！")
