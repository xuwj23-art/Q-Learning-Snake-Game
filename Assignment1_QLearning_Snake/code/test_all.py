"""
测试脚本 - 确保所有代码正常工作
Test Script to Verify All Code Works
"""

import sys
import os

print("=" * 70)
print("Q-Learning贪吃蛇项目 - 完整测试")
print("=" * 70)

# 测试1: 检查文件是否存在
print("\n[测试1] 检查文件结构...")
files_to_check = [
    'snake_game.py',
    'q_learning_agent.py',
    'train.py',
    'visualize.py',
    'requirements.txt'
]

all_files_exist = True
for file in files_to_check:
    exists = os.path.exists(file)
    status = "✓" if exists else "✗"
    print(f"  {status} {file}")
    if not exists:
        all_files_exist = False

if not all_files_exist:
    print("\n❌ 部分文件缺失！请确保所有文件都在code目录中。")
    sys.exit(1)

print("\n✅ 所有文件都存在！")

# 测试2: 导入模块
print("\n[测试2] 测试模块导入...")
try:
    from snake_game import SnakeGameEnv, Direction, Point
    print("  ✓ snake_game 模块导入成功")
except Exception as e:
    print(f"  ✗ snake_game 导入失败: {e}")
    sys.exit(1)

try:
    from q_learning_agent import QLearningAgent
    print("  ✓ q_learning_agent 模块导入成功")
except Exception as e:
    print(f"  ✗ q_learning_agent 导入失败: {e}")
    sys.exit(1)

print("\n✅ 所有模块导入成功！")

# 测试3: 游戏环境
print("\n[测试3] 测试游戏环境...")
try:
    env = SnakeGameEnv(width=10, height=10)
    state = env.reset()
    print(f"  ✓ 环境初始化成功")
    print(f"  - 状态维度: {len(state)}")
    print(f"  - 初始状态: {state}")
    
    # 执行几步
    for i in range(3):
        action = [1, 0, 0]  # 直行
        state, reward, done, score = env.step(action)
        print(f"  - 步骤{i+1}: reward={reward:.2f}, done={done}, score={score}")
        if done:
            break
    
    print("  ✓ 游戏环境测试通过")
except Exception as e:
    print(f"  ✗ 游戏环境测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ 游戏环境工作正常！")

# 测试4: Q-Learning智能体
print("\n[测试4] 测试Q-Learning智能体...")
try:
    agent = QLearningAgent(state_size=11, action_size=3)
    print(f"  ✓ 智能体初始化成功")
    print(f"  - 学习率: {agent.learning_rate}")
    print(f"  - 折扣因子: {agent.discount_factor}")
    print(f"  - 初始Epsilon: {agent.epsilon}")
    
    # 测试动作选择
    test_state = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]
    action = agent.get_action(test_state)
    print(f"  ✓ 动作选择成功: {action}")
    
    # 测试Q表更新
    next_state = [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1]
    reward = 1.0
    agent.update(test_state, action, reward, next_state, done=False)
    print(f"  ✓ Q表更新成功")
    print(f"  - Q表大小: {agent.get_q_table_size()}")
    
except Exception as e:
    print(f"  ✗ Q-Learning智能体测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ Q-Learning智能体工作正常！")

# 测试5: 简短训练测试
print("\n[测试5] 测试训练流程（10轮快速测试）...")
try:
    env = SnakeGameEnv(width=10, height=10)
    agent = QLearningAgent()
    
    scores = []
    for episode in range(1, 11):
        state = env.reset()
        done = False
        
        while not done:
            action = agent.get_action(state, training=True)
            next_state, reward, done, score = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
        
        agent.decay_epsilon()
        scores.append(score)
        
        if episode % 5 == 0:
            print(f"  - Episode {episode}: score={score}, epsilon={agent.epsilon:.3f}")
    
    print(f"  ✓ 训练流程测试通过")
    print(f"  - 平均得分: {sum(scores)/len(scores):.2f}")
    print(f"  - Q表大小: {agent.get_q_table_size()}")
    
except Exception as e:
    print(f"  ✗ 训练流程测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ 训练流程工作正常！")

# 测试6: 检查依赖包
print("\n[测试6] 检查依赖包...")
required_packages = {
    'numpy': 'numpy',
    'matplotlib': 'matplotlib.pyplot',
    'pygame': 'pygame'
}

missing_packages = []
for package_name, import_name in required_packages.items():
    try:
        __import__(import_name)
        print(f"  ✓ {package_name} 已安装")
    except ImportError:
        print(f"  ✗ {package_name} 未安装")
        missing_packages.append(package_name)

if missing_packages:
    print(f"\n⚠️  缺少依赖包: {', '.join(missing_packages)}")
    print("请运行: pip install -r requirements.txt")
else:
    print("\n✅ 所有依赖包都已安装！")

# 最终总结
print("\n" + "=" * 70)
print("测试完成总结")
print("=" * 70)

print("\n✅ 所有核心功能测试通过！")
print("\n项目已准备就绪，可以进行以下操作：")
print("  1. 运行完整训练: python train.py")
print("  2. 可视化演示: python visualize.py agent")
print("  3. 性能对比: python visualize.py compare")

print("\n下一步工作：")
print("  [ ] 运行完整训练（500轮）")
print("  [ ] 记录训练结果数据")
print("  [ ] 填写报告模板")
print("  [ ] 录制演示视频")
print("  [ ] 上传到GitHub")
print("  [ ] 提交到Moodle")

print("\n" + "=" * 70)
print("祝你顺利完成作业！🎉")
print("=" * 70)
