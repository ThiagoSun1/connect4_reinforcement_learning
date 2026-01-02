# Connect 4 Reinforcement learning

A Jetson Orin Nano-based you vs. a AI at connect 4 **uses reinforcement learning to train an AI to play against you in the game of connect 4** and visualizes it using Pygame.

---

##  🔴🟡 Overview

This project allows you to **play a live game of connect 4 against an AI**. The system starts with your turn, and then lets the AI pick its move.

---

## 🧠 Features

- 🎯 Uses Torch to train a model to play against you.
- 🎮 Plays a live game using Pygame
- ⚙️ Runs entirely on a Jetson Orin Nano
- 🐍 Written in pure Python with no external apps or software needed

---

## 📦 Hardware Requirements

- Jetson Orin Nano
  
---

## 🧰 Software Requirements

Before installing anything, make sure you're in a virtual enviroment:

```bash

# clone the repository
git clone https://github.com/ThiagoSun1/connect4_reinforcement_learning

# install package to create virtual enviroment
sudo apt install python3-venv

# make virtual enviroment
python3 -m venv rl_venv

# activate virtual enviroment
source ~/rl_venv/bin/activate

```

- Python 3.10 (pre-installed on Jetson Orin Nano in virtual envirments)
- The following Python libraries:

```bash
sudo apt update
sudo apt install python3-pip

# install necessary dependencies
pip install -r requirements.txt --force-reinstall

# run the tracker
cd ~/connect4_reinforcement_learning
python3 connect4_extend.py

```
