"""
快速测试Pygame环境
Quick Test for Pygame Environment
"""

import sys
import os

print("=" * 60)
print("Pygame环境测试")
print("=" * 60)

# 测试1: 检查pygame
print("\n[测试1] 检查pygame安装...")
try:
    import pygame
    print(f"  ✓ Pygame已安装: 版本 {pygame.__version__}")
except ImportError:
    print("  ✗ Pygame未安装")
    print("  请运行: pip install pygame")
    sys.exit(1)

# 测试2: 初始化pygame
print("\n[测试2] 初始化pygame...")
try:
    pygame.init()
    print("  ✓ Pygame初始化成功")
except Exception as e:
    print(f"  ✗ Pygame初始化失败: {e}")
    sys.exit(1)

# 测试3: 创建测试窗口
print("\n[测试3] 创建测试窗口（3秒后自动关闭）...")
try:
    # 创建小窗口
    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption('Pygame测试窗口')
    
    # 填充颜色
    screen.fill((0, 128, 0))  # 绿色背景
    
    # 显示文字
    font = pygame.font.Font(None, 36)
    text = font.render('Pygame Works!', True, (255, 255, 255))
    text_rect = text.get_rect(center=(200, 150))
    screen.blit(text, text_rect)
    
    pygame.display.flip()
    
    print("  ✓ 窗口创建成功！")
    print("  如果你看到一个绿色窗口，说明Pygame工作正常")
    print("  窗口将在3秒后自动关闭...")
    
    # 等待3秒
    pygame.time.wait(3000)
    
    pygame.quit()
    
except Exception as e:
    print(f"  ✗ 窗口创建失败: {e}")
    print("  可能原因：")
    print("    - 在远程环境或SSH中运行")
    print("    - 没有图形界面")
    print("    - 显示驱动问题")
    sys.exit(1)

# 测试4: 检查项目文件
print("\n[测试4] 检查项目文件...")
required_files = [
    'snake_game.py',
    'q_learning_agent.py',
    'visualize.py',
    '../results/q_learning_snake_best.pkl'
]

all_exist = True
for file in required_files:
    exists = os.path.exists(file)
    status = "✓" if exists else "✗"
    print(f"  {status} {file}")
    if not exists:
        all_exist = False

if not all_exist:
    print("\n  ⚠️ 部分文件缺失")
    print("  如果缺少模型文件，请先运行: python train.py")

print("\n" + "=" * 60)
print("✅ 环境测试完成！")
print("\n你可以运行以下命令：")
print("  python visualize.py agent    # 观看智能体玩游戏")
print("  python visualize.py random   # 观看随机策略")
print("  python visualize.py compare  # 性能对比测试")
print("=" * 60)
