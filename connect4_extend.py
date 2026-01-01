import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import os
import signal
import sys

# ------------------ CONSTANTS ------------------
ROWS, COLS = 6, 7
STATE_SIZE = ROWS * COLS
MINIMAX_DEPTH = 6

# ------------------ GAME ENVIRONMENT ------------------
class ConnectFour:
    def __init__(self):
        self.board = np.zeros((ROWS, COLS), dtype=int)
        self.current_player = 1  # 1 = AI (maximizing), -1 = opponent

    def reset(self):
        self.board[:] = 0
        self.current_player = 1
        return self.board.flatten().copy()

    def valid_actions(self):
        return [c for c in range(COLS) if self.board[0][c] == 0]

    def step(self, col):
        valid = self.valid_actions()
        if col not in valid:
            return self.board.flatten().copy(), -1.0, True

        # Drop piece
        for r in reversed(range(ROWS)):
            if self.board[r][col] == 0:
                self.board[r][col] = self.current_player
                break

        winner = self.check_winner()
        done = winner != 0 or len(self.valid_actions()) == 0
        reward = -0.01  # Small step penalty

        if winner == 1:
            reward = 1.0
        elif winner == -1:
            reward = -1.0

        self.current_player *= -1
        return self.board.flatten().copy(), reward, done

    # ------------------ WIN CHECK ------------------
    def check_winner(self):
        return self.check_winner_board(self.board)

    def check_winner_board(self, board):
        # Horizontal
        for r in range(ROWS):
            for c in range(COLS - 3):
                if board[r][c] != 0 and all(board[r][c+i] == board[r][c] for i in range(4)):
                    return board[r][c]
        # Vertical
        for c in range(COLS):
            for r in range(ROWS - 3):
                if board[r][c] != 0 and all(board[r+i][c] == board[r][c] for i in range(4)):
                    return board[r][c]
        # Diagonal down-right
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                if board[r][c] != 0 and all(board[r+i][c+i] == board[r][c] for i in range(4)):
                    return board[r][c]
        # Diagonal down-left
        for r in range(ROWS - 3):
            for c in range(3, COLS):
                if board[r][c] != 0 and all(board[r+i][c-i] == board[r][c] for i in range(4)):
                    return board[r][c]
        return 0

    # ------------------ STRATEGIC HELPERS ------------------
    def drop_piece_temp(self, col, player):
        for r in reversed(range(ROWS)):
            if self.board[r][col] == 0:
                self.board[r][col] = player
                return r
        return -1

    def undo_move(self, row, col):
        if row >= 0:
            self.board[row][col] = 0

    def has_winning_move(self, player):
        for col in self.valid_actions():
            row = self.drop_piece_temp(col, player)
            if self.check_winner() == player:
                self.undo_move(row, col)
                return True, col
            self.undo_move(row, col)
        return False, None

    def creates_fork(self, col, player=1):
        row = self.drop_piece_temp(col, player)
        threats = 0
        for c in self.valid_actions():
            if c == col and self.board[0][col] != 0:
                continue
            r2 = self.drop_piece_temp(c, player)
            if self.check_winner() == player:
                threats += 1
            self.undo_move(r2, c)
        self.undo_move(row, col)
        return threats >= 2, []

    # ------------------ MINIMAX ------------------
    def minimax(self, depth, alpha, beta, maximizing_player):
        winner = self.check_winner()
        valid = self.valid_actions()

        if winner == 1:
            return 1000000 + depth, None
        elif winner == -1:
            return -1000000 - depth, None
        elif not valid:
            return 0, None
        elif depth == 0:
            return self.evaluate_position(), None

        best_col = valid[0]
        if maximizing_player:
            max_eval = float('-inf')
            for col in valid:
                row = self.drop_piece_temp(col, 1)
                eval_score, _ = self.minimax(depth - 1, alpha, beta, False)
                self.undo_move(row, col)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_col = col
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval, best_col
        else:
            min_eval = float('inf')
            for col in valid:
                row = self.drop_piece_temp(col, -1)
                eval_score, _ = self.minimax(depth - 1, alpha, beta, True)
                self.undo_move(row, col)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_col = col
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval, best_col

    def evaluate_position(self):
        score = 0
        center_col = COLS // 2
        center_count = np.sum(self.board[:, center_col] == 1)
        score += center_count * 4

        def evaluate_window(window):
            ai = window.count(1)
            opp = window.count(-1)
            empty = 4 - ai - opp
            if ai == 4: return 100
            if opp == 4: return -100
            if ai == 3 and empty == 1: return 10
            if opp == 3 and empty == 1: return -8
            if ai == 2 and empty == 2: return 3
            if opp == 2 and empty == 2: return -2
            return 0

        for r in range(ROWS):
            for c in range(COLS - 3):
                score += evaluate_window(self.board[r, c:c+4].tolist())
        for c in range(COLS):
            for r in range(ROWS - 3):
                score += evaluate_window(self.board[r:r+4, c].tolist())
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                score += evaluate_window([self.board[r+i][c+i] for i in range(4)])
        for r in range(ROWS - 3):
            for c in range(3, COLS):
                score += evaluate_window([self.board[r+i][c-i] for i in range(4)])

        return score

    def get_best_move_strategic(self):
        valid = self.valid_actions()
        if not valid:
            return None

        win_possible, win_col = self.has_winning_move(1)
        if win_possible:
            return win_col

        opp_win_possible, opp_win_col = self.has_winning_move(-1)
        if opp_win_possible:
            return opp_win_col

        for col in valid:
            creates, _ = self.creates_fork(col, 1)
            if creates:
                return col

        for col in valid:
            row = self.drop_piece_temp(col, -1)
            creates, _ = self.creates_fork(col, -1)
            self.undo_move(row, col)
            if creates:
                return col

        _, best_col = self.minimax(MINIMAX_DEPTH, float('-inf'), float('inf'), True)
        return best_col if best_col is not None else random.choice(valid)

# ------------------ DQN (unchanged architecture) ------------------
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, COLS)
        )

    def forward(self, x):
        return self.net(x)

# ------------------ TRAIN STEP WITH AMP ------------------
def train_step(model, target, memory, optimizer, scaler, batch_size, gamma, device):
    if len(memory) < batch_size:
        return
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
    actions = torch.tensor(actions, device=device).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.float32, device=device)

    with torch.amp.autocast(device_type='cuda'):
        q_values = model(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_q = target(next_states).max(1)[0]
            target_q = rewards + gamma * next_q * (1 - dones)

        loss = nn.SmoothL1Loss()(q_values, target_q)

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# ------------------ MAIN ------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    env = ConnectFour()
    model = DQN().to(device)
    target = DQN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler()  # For mixed precision
    memory = deque(maxlen=100_000)

    # ==================== RESUME OR FRESH START ====================
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    resume_from = None  # Set to resume, e.g., "checkpoints/connect4_checkpoint_50000.pth"

    start_episode = 1
    epsilon = 1.0
    episode = 0

    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_episode = checkpoint['episode'] + 1
        epsilon = checkpoint.get('epsilon', 0.05)
        print(f"Resumed at episode {start_episode}, epsilon = {epsilon:.3f}")
    else:
        print("Starting fresh training...")

    target.load_state_dict(model.state_dict())

    # ==================== CTRL+C SAFE SAVE ====================
    def save_emergency_checkpoint():
        print("\n\nCtrl+C detected! Saving emergency checkpoint...")
        emergency_path = os.path.join(checkpoint_dir, f"connect4_emergency_{episode}.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'episode': episode,
            'epsilon': epsilon
        }, emergency_path)
        print(f"Emergency checkpoint saved: {emergency_path}")
        print("You can resume by setting resume_from to this file path.")

    def signal_handler(sig, frame):
        save_emergency_checkpoint()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    print("Ctrl+C safe shutdown enabled — will save emergency checkpoint on interrupt.")
    # ==========================================================

    total_episodes = 200_000
    batch_size = 256          # Larger batch for better GPU utilization
    gamma = 0.99
    epsilon_min = 0.05
    epsilon_decay = 0.997
    target_update = 1000
    checkpoint_interval = 10000
    use_strategic = False

    for episode in range(start_episode, total_episodes + 1):
        state = env.reset()
        done = False

        while not done:
            valid = env.valid_actions()
            if not valid:
                break

            if random.random() < epsilon:
                action = random.choice(valid)
            else:
                if use_strategic:
                    action = env.get_best_move_strategic()
                else:
                    with torch.no_grad():
                        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                        q_values = model(state_tensor)[0]
                        mask = torch.full((COLS,), -1e9, device=device)
                        for c in valid:
                            mask[c] = 0
                        q_values = q_values + mask
                        action = torch.argmax(q_values).item()

            next_state, reward, done = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state

            train_step(model, target, memory, optimizer, scaler, batch_size, gamma, device)

            if done:
                break

            opp_valid = env.valid_actions()
            if opp_valid:
                opp_action = random.choice(opp_valid)
                state, _, done = env.step(opp_action)

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if episode % target_update == 0:
            target.load_state_dict(model.state_dict())

        if episode % 5000 == 0:
            print(f"Episode {episode}/{total_episodes} | Epsilon: {epsilon:.4f} | Memory: {len(memory)}")

        # Regular checkpoint
        if episode % checkpoint_interval == 0 or episode == total_episodes:
            checkpoint_path = os.path.join(checkpoint_dir, f"connect4_checkpoint_{episode}.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'episode': episode,
                'epsilon': epsilon
            }, checkpoint_path)
            print(f"Regular checkpoint saved: {checkpoint_path}")

    final_path = "connect4_dqn_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"Training complete! Final model saved as {final_path}")
