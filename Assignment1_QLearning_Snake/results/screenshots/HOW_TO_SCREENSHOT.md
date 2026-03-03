# 📸 Screenshot Instructions

## What You Need to Do RIGHT NOW

### Step 1: Run the game
```cmd
cd D:\Document\香港学习\ML\Assignment1_QLearning_Snake\code
python visualize.py agent
```

### Step 2: Capture 3 screenshots

#### Screenshot 1: Initial State (0-2 seconds)
- **When**: Right when game starts
- **What to show**: Snake at starting position, first food appears
- **How**: Press `Win + Shift + S`, drag to capture window
- **Save as**: `screenshot_initial_state.png`
- **Save to**: `D:\Document\香港学习\ML\Assignment1_QLearning_Snake\results\screenshots\`

#### Screenshot 2: Playing (10-15 seconds)
- **When**: Snake length is 5-8
- **What to show**: Snake moving, chasing food, score around 3-5
- **How**: Press `Win + Shift + S`, capture
- **Save as**: `screenshot_playing.png`
- **Save to**: Same folder

#### Screenshot 3: High Score (when it happens)
- **When**: Snake gets longest (score 10+) OR before game over
- **What to show**: Long snake, high score in info panel
- **How**: Press `Win + Shift + S`, capture quickly!
- **Save as**: `screenshot_high_score.png`
- **Save to**: Same folder

### Step 3: Code screenshot

1. Open `q_learning_agent.py` in VSCode or Notepad
2. Scroll to lines 40-60 (the `update` function)
3. Select the entire function
4. Press `Win + Shift + S`, capture the code
5. Save as `code_q_learning_update.png` in screenshots folder

---

## Quick Screenshot Tips

- Use **Win + Shift + S** (Windows Snipping Tool)
- Capture the ENTIRE Pygame window including the title bar
- Make sure info panel on right is visible
- Save as PNG format
- Don't worry about perfect timing - just get something!

---

## After Screenshots

All 4 files should be in:
```
D:\Document\香港学习\ML\Assignment1_QLearning_Snake\results\screenshots\
├── screenshot_initial_state.png
├── screenshot_playing.png
├── screenshot_high_score.png
└── code_q_learning_update.png
```

Then your report will display all images correctly! ✅

---

## If You Miss a Moment

Just run `python visualize.py agent` again!
The game runs multiple episodes automatically.
