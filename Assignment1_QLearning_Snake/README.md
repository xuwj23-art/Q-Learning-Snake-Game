# Q-Learning Snake Game

A reinforcement learning project implementing Q-Learning algorithm to train an agent to play the classic Snake game.

**Course**: CDS524 - Machine Learning  
**Demo Video**: [YouTube Link - To be added]  
**GitHub**: https://github.com/xuwj23-art/Q-Learning-Snake-Game

---

## 🎮 Project Overview

This project demonstrates the application of Q-Learning reinforcement learning to create an intelligent agent that learns to play Snake game through trial and error. The agent improves its performance from random play to achieving consistently high scores.

### Key Features

- **Complete Q-Learning Implementation**: Epsilon-greedy exploration, learning rate, discount factor
- **11-Dimensional State Space**: Efficient representation capturing danger, direction, and food location
- **Interactive Visualization**: Real-time game display using Pygame
- **Performance Analysis**: Detailed training curves and comparison metrics

---

## 📊 Results

- **Highest Score**: 22
- **Average Score**: 6.66 (from 0.37 in first 100 episodes)
- **Performance Improvement**: 1340% over random strategy
- **Training Time**: 0.37 seconds for 500 episodes
- **Q-Table Size**: 215 unique states

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install numpy matplotlib pygame
```

### Running the Project

```bash
# Train the agent
cd code
python train.py

# Watch the trained agent play
python visualize.py agent

# Compare with random strategy
python visualize.py random
```

---

## 📁 Project Structure

```
Q-Learning-Snake-Game/
│
├── code/
│   ├── snake_game.py          # Game environment
│   ├── q_learning_agent.py    # Q-Learning algorithm
│   ├── train.py               # Training script
│   ├── visualize.py           # Pygame visualization
│   └── requirements.txt       # Dependencies
│
├── notebook/
│   └── colab_qlearning_snake.py  # Google Colab version
│
├── results/
│   ├── training_results.png   # Training curves
│   └── screenshots/           # Game UI screenshots
│
├── report/
│   └── Report_Final.md        # Project report (1450 words)
│
└── README.md
```

---

## 🎯 Game Design

### State Space (11 dimensions)
- **Danger Detection** (3): Front, right, left
- **Movement Direction** (4): Left, right, up, down
- **Food Location** (4): Food relative position

### Action Space (3 actions)
- **Straight**: Continue current direction
- **Right Turn**: Turn 90° clockwise
- **Left Turn**: Turn 90° counter-clockwise

### Reward Function
- `+10`: Eat food
- `-10`: Game over (collision)
- `-0.01`: Each step (encourage efficiency)

---

## 🧠 Q-Learning Algorithm

### Core Parameters
- **Learning Rate (α)**: 0.1
- **Discount Factor (γ)**: 0.95
- **Initial Epsilon (ε)**: 1.0
- **Epsilon Decay**: 0.995
- **Minimum Epsilon**: 0.01

### Update Formula

```
Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
```

---

## 📈 Training Results

The agent shows clear learning progression:

1. **Episodes 1-100**: Exploration phase, average score 0.37
2. **Episodes 100-300**: Rapid improvement, reaching 2.31 average
3. **Episodes 300-500**: Refinement, achieving 6.66 average

See `results/training_results.png` for detailed visualization.

---

## 🎬 Demo

Watch the agent in action: [YouTube Demo Link - To be added]

The video demonstrates:
- Game mechanics
- Q-Learning algorithm explanation
- Training process visualization
- Performance comparison (random vs. trained)

---

## 🔧 Technical Implementation

### Environment (`snake_game.py`)
- Grid-based game logic
- State representation
- Reward calculation
- Collision detection

### Agent (`q_learning_agent.py`)
- Q-table using dictionary
- Epsilon-greedy policy
- Q-value updates
- Model save/load

### Training (`train.py`)
- Training loop
- Progress tracking
- Model checkpointing
- Results visualization

### Visualization (`visualize.py`)
- Pygame interface
- Real-time display
- Information panel
- Multiple demo modes

---

## 📝 Documentation

- **Report**: See `report/Report_Final.md` (1000-1500 words)
- **Video Script**: See `VIDEO_SCRIPT.md`
- **Google Colab**: See `notebook/colab_qlearning_snake.py`

---

## 🏆 Performance Comparison

| Metric | Random Strategy | Q-Learning Agent | Improvement |
|--------|----------------|------------------|-------------|
| Average Score | 0.5 ± 0.8 | 7.2 ± 4.5 | **+1340%** |
| Highest Score | 3 | 22 | **+633%** |
| Success Rate | ~15% | ~75% | **+400%** |

---

## 💡 Key Learnings

### Challenges Faced
1. **Sparse Rewards**: Solved with step penalty
2. **State Design**: Balanced information vs. Q-table size
3. **Exploration vs. Exploitation**: Tuned epsilon decay

### Solutions Implemented
- Compact 11-D state representation
- Carefully designed reward structure
- Exponential epsilon decay strategy

---

## 🔮 Future Improvements

- Implement Deep Q-Network (DQN) for larger grids
- Add experience replay mechanism
- Test on different grid sizes
- Compare with other RL algorithms (SARSA, A3C)

---

## 📚 References

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*
2. Watkins, C. J., & Dayan, P. (1992). Q-learning. *Machine Learning*, 8(3-4), 279-292
3. [Q-Learning Algorithm](https://en.wikipedia.org/wiki/Q-learning)
4. [Pygame Documentation](https://www.pygame.org/docs/)

---

## 📧 Contact

**GitHub**: https://github.com/xuwj23-art/Q-Learning-Snake-Game  
**Course**: CDS524 - Machine Learning  
**Date**: March 3, 2026

---

## 📜 License

This project is for educational purposes as part of CDS524 course assignment.

---

## 🙏 Acknowledgments

- Course Instructor and TAs
- Reference projects: [QlearningTankWar](https://github.com/KI-cheng/QlearningTankWar)
- Pygame and Python communities

---

**⭐ If you find this project helpful, please star the repository!**
