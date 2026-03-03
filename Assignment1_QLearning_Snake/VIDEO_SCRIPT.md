# 🎬 Video Recording Script - Q-Learning Snake Game

**Duration**: 3-5 minutes  
**Language**: English  
**Style**: Professional but conversational

---

## 🎥 SCENE 1: Introduction (30 seconds)

### Visual:
- Show project title slide or code directory

### Script:
```
Hello! Today I'm presenting my Q-Learning Snake Game project for CDS524.

In this project, I implemented a reinforcement learning agent that learns to play 
the classic snake game. The agent uses Q-Learning algorithm to discover the best 
strategy through trial and error.

Let me show you how it works.
```

---

## 🎥 SCENE 2: Project Overview (30 seconds)

### Visual:
- Show folder structure or README

### Script:
```
The project consists of four main components:

First, the game environment in snake_game.py - this is a 10 by 10 grid where the 
snake moves around.

Second, the Q-Learning agent in q_learning_agent.py - this is the brain that 
learns from experience.

Third, the training script that teaches the agent.

And finally, the visualization using Pygame to see the agent in action.

Let's start by looking at the game rules.
```

---

## 🎥 SCENE 3: Game Design (45 seconds)

### Visual:
- Show game running (initial state)
- Point to different elements

### Script:
```
Here's the game interface. The green squares are the snake, and the red circle 
is the food.

The goal is simple: eat as much food as possible without hitting the walls or 
the snake's own body.

The state space has 11 dimensions. It captures three things: danger detection 
in three directions, the current moving direction, and where the food is located 
relative to the snake's head.

The agent can choose from three actions: go straight, turn left, or turn right.

The reward system is straightforward: plus 10 for eating food, minus 10 for 
dying, and a small negative reward each step to encourage efficiency.
```

---

## 🎥 SCENE 4: Q-Learning Algorithm (60 seconds)

### Visual:
- Show code (q_learning_agent.py update function)
- Highlight key parts

### Script:
```
Now let's look at the Q-Learning implementation.

Here's the core update function. It uses the classic Q-Learning formula: 
Q of s, a gets updated by adding the learning rate times the difference between 
the target Q-value and the current Q-value.

We use epsilon-greedy strategy to balance exploration and exploitation. 
At the start, epsilon is 1.0, so the agent explores randomly. As training 
progresses, epsilon decays to 0.01, and the agent starts exploiting what it 
has learned.

The learning rate is 0.1, which means we update gradually and carefully.
The discount factor is 0.95, so we care about long-term rewards.

Let me show you the training results.
```

---

## 🎥 SCENE 5: Training Results (60 seconds)

### Visual:
- Show training_results.png (4 panels)
- Point to each graph

### Script:
```
This chart shows the training process over 500 episodes.

The top-left shows the score over time. You can see it starts near zero and 
gradually improves to an average of 6.66. The best score reached was 22!

The top-right shows epsilon decay. It smoothly drops from 1.0 to about 0.08, 
which shows the transition from exploration to exploitation.

The bottom-left shows Q-table growth. The agent explored 215 unique states 
during training.

And the bottom-right shows the score distribution for the last 100 games. 
Most games score between 4 and 10, which is much better than random play.

The entire training took only 0.37 seconds on my computer.
```

---

## 🎥 SCENE 6: Performance Comparison (45 seconds)

### Visual:
- First show random strategy playing (5-10 seconds)
- Then show trained agent playing (15-20 seconds)

### Script:
```
Let me show you the difference between random strategy and the trained agent.

First, here's random strategy. Watch how the snake moves aimlessly and quickly 
crashes into walls or itself. Random strategy averages only about 0.5 points.

Now, here's the trained Q-Learning agent. Notice how it actively seeks the food 
and avoids obstacles. It makes smart decisions about when to turn and when to 
go straight. The trained agent averages 7.2 points - that's a 1340% improvement!

You can see it's learned effective strategies like creating safe paths and 
planning ahead.
```

---

## 🎥 SCENE 7: Challenges and Solutions (30 seconds)

### Visual:
- Show report or slides with challenge points

### Script:
```
During development, I faced three main challenges.

First, the sparse reward problem. The agent rarely found food early on. 
I solved this by adding a small step penalty to encourage movement.

Second, designing the state space. I had to balance between having enough 
information and keeping the Q-table size manageable. The 11-dimensional state 
proved to be the sweet spot.

Third, balancing exploration and exploitation. The exponential epsilon decay 
worked perfectly - fast enough to converge but slow enough to explore thoroughly.
```

---

## 🎥 SCENE 8: Live Demonstration (30 seconds)

### Visual:
- Show agent playing multiple games
- Let it run for 20-30 seconds

### Script:
```
Let me run a few more games so you can see the agent in action.

Watch how it navigates the grid... There it goes, eating food... growing longer... 
avoiding the walls... making smart turns...

The agent isn't perfect - it still makes mistakes sometimes - but it's learned 
a solid strategy that works well most of the time.

This is the power of reinforcement learning: the agent discovered these strategies 
entirely on its own through experience.
```

---

## 🎥 SCENE 9: Conclusion (20 seconds)

### Visual:
- Show final results summary or thank you slide

### Script:
```
To summarize: I successfully implemented a Q-Learning agent that learned to play 
snake game, achieving 22 as the highest score and an average of 6.66 points.

The project demonstrates key reinforcement learning concepts: state representation, 
reward design, and the exploration-exploitation tradeoff.

Thank you for watching! Feel free to check out the code on GitHub.
```

---

## 📋 Recording Checklist

**Before Recording:**
- [ ] Test visualize.py runs smoothly
- [ ] Have training_results.png open
- [ ] Have code files ready to show
- [ ] Close unnecessary programs
- [ ] Prepare recording software (Win+G)

**During Recording:**
- [ ] Speak clearly and at moderate pace
- [ ] Pause briefly between sections
- [ ] Point to screen elements when describing them
- [ ] Show enthusiasm but stay professional

**After Recording:**
- [ ] Review for audio clarity
- [ ] Check if all demos are visible
- [ ] Trim any awkward pauses
- [ ] Add title slide if needed

---

## 🎯 Pro Tips

1. **Practice the script 2-3 times** before recording
2. **Don't worry about being perfect** - authentic is better than robotic
3. **Use natural pauses** - it's okay to pause while code runs
4. **Smile while speaking** - it makes your voice sound friendlier
5. **Keep energy up** - stay engaged throughout

---

## ⏱️ Timing Guide

| Section | Duration | Cumulative |
|---------|----------|------------|
| Introduction | 30s | 0:30 |
| Overview | 30s | 1:00 |
| Game Design | 45s | 1:45 |
| Q-Learning | 60s | 2:45 |
| Training Results | 60s | 3:45 |
| Performance Comparison | 45s | 4:30 |
| Challenges | 30s | 5:00 |
| Live Demo | 30s | 5:30 |
| Conclusion | 20s | 5:50 |

**Total: 5 minutes 50 seconds** (can trim to 5 minutes by shortening live demo)

---

Good luck with your recording! 🎬
