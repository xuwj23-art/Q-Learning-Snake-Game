# 简化版游戏界面生成器 - 可在Colab运行
# 复制此代码到Google Colab，然后点击运行

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 手动创建几个游戏状态示例

def draw_snake_game(snake_positions, food_position, width=10, height=10, title="Snake Game", score=0):
    """
    绘制贪吃蛇游戏界面
    
    参数:
        snake_positions: 蛇的位置列表 [(x1,y1), (x2,y2), ...]
        food_position: 食物位置 (x, y)
        width: 游戏宽度
        height: 游戏高度
        title: 标题
        score: 得分
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # 设置坐标轴
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, height - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    
    # 绘制网格
    for i in range(width + 1):
        ax.axvline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
    for i in range(height + 1):
        ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
    
    # 绘制蛇
    for i, (x, y) in enumerate(snake_positions):
        if i == 0:  # 蛇头
            color = 'darkgreen'
            edgecolor = 'black'
            linewidth = 2
        else:  # 蛇身
            color = 'lime'
            edgecolor = 'darkgreen'
            linewidth = 1
        
        rect = patches.Rectangle((x - 0.4, y - 0.4), 0.8, 0.8,
                                 linewidth=linewidth, edgecolor=edgecolor, facecolor=color)
        ax.add_patch(rect)
    
    # 绘制食物
    food_circle = patches.Circle(food_position, 0.3,
                                 linewidth=2, edgecolor='darkred', facecolor='red')
    ax.add_patch(food_circle)
    
    # 添加信息
    info_text = f"Score: {score} | Snake Length: {len(snake_positions)}"
    ax.text(width / 2, -1.2, info_text, ha='center', fontsize=12, weight='bold')
    
    ax.set_title(title, fontsize=14, weight='bold', pad=20)
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    
    plt.tight_layout()
    return fig

# 示例1: 初始状态
print("生成图1: 初始状态")
snake1 = [(5, 5), (4, 5), (3, 5)]  # 蛇头在(5,5)
food1 = (7, 3)
fig1 = draw_snake_game(snake1, food1, title="Fig 2.1 - Initial Game State", score=0)
plt.savefig('screenshot_1_initial.png', dpi=150, bbox_inches='tight')
plt.show()

# 示例2: 游戏进行中
print("生成图2: 游戏进行中")
snake2 = [(6, 3), (5, 3), (4, 3), (3, 3), (2, 3)]
food2 = (8, 7)
fig2 = draw_snake_game(snake2, food2, title="Fig 2.2 - Game in Progress", score=2)
plt.savefig('screenshot_2_playing.png', dpi=150, bbox_inches='tight')
plt.show()

# 示例3: 高分状态
print("生成图3: 高分状态")
snake3 = [
    (5, 5), (6, 5), (7, 5), (7, 6), (7, 7), 
    (6, 7), (5, 7), (4, 7), (3, 7), (3, 6),
    (3, 5), (3, 4)
]
food3 = (1, 1)
fig3 = draw_snake_game(snake3, food3, title="Fig 2.3 - High Score Game", score=9)
plt.savefig('screenshot_3_highscore.png', dpi=150, bbox_inches='tight')
plt.show()

# 对比图
print("生成对比图")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 随机策略（短蛇）
snake_random = [(2, 2), (1, 2), (0, 2)]
food_random = (8, 8)

ax = axes[0]
ax.set_xlim(-0.5, 9.5)
ax.set_ylim(-0.5, 9.5)
ax.set_aspect('equal')
ax.invert_yaxis()

for i in range(11):
    ax.axvline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
    ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)

for i, (x, y) in enumerate(snake_random):
    color = 'darkgreen' if i == 0 else 'lime'
    rect = patches.Rectangle((x - 0.4, y - 0.4), 0.8, 0.8, facecolor=color, edgecolor='black')
    ax.add_patch(rect)

ax.add_patch(patches.Circle(food_random, 0.3, facecolor='red', edgecolor='darkred', linewidth=2))
ax.set_title('Random Strategy (Score: 0)', fontsize=12, weight='bold')

# Q-Learning（长蛇）
snake_ql = [(5, 5), (6, 5), (7, 5), (7, 6), (7, 7), (6, 7), (5, 7)]
food_ql = (2, 2)

ax = axes[1]
ax.set_xlim(-0.5, 9.5)
ax.set_ylim(-0.5, 9.5)
ax.set_aspect('equal')
ax.invert_yaxis()

for i in range(11):
    ax.axvline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
    ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)

for i, (x, y) in enumerate(snake_ql):
    color = 'darkgreen' if i == 0 else 'lime'
    rect = patches.Rectangle((x - 0.4, y - 0.4), 0.8, 0.8, facecolor=color, edgecolor='black')
    ax.add_patch(rect)

ax.add_patch(patches.Circle(food_ql, 0.3, facecolor='red', edgecolor='darkred', linewidth=2))
ax.set_title('Q-Learning Agent (Score: 4)', fontsize=12, weight='bold')

plt.tight_layout()
plt.savefig('screenshot_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ 所有截图已生成！")
print("下载文件:")
print("  - screenshot_1_initial.png")
print("  - screenshot_2_playing.png")
print("  - screenshot_3_highscore.png")
print("  - screenshot_comparison.png")
