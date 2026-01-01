import pygame
import sys
import torch
import numpy as np

# ------------------ IMPORT YOUR CLASSES ------------------
# Make sure this file has the correct DQN and ConnectFour classes
from connect4_extend import DQN, ConnectFour

# ------------------ PYGAME SETUP ------------------
pygame.init()

BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)

ROWS = 6
COLS = 7
SQUARESIZE = 100
RADIUS = int(SQUARESIZE / 2 - 5)

width = COLS * SQUARESIZE
height = (ROWS + 1) * SQUARESIZE
size = (width, height)
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Connect 4 - Human vs Trained DQN AI")

font_large = pygame.font.SysFont("monospace", 75)
font_small = pygame.font.SysFont("monospace", 35)

# ------------------ LOAD TRAINED MODEL ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading model on {device}...")

model = DQN().to(device)
model.load_state_dict(torch.load("connect4_dqn.pth", map_location=device))
model.eval()
print("Model loaded successfully!")

# ------------------ GAME SETUP ------------------
env = ConnectFour()
env.reset()

game_over = False
turn = 0  # 0 = human (red), 1 = AI (yellow)
hover_col = -1

# ------------------ DRAW BOARD ------------------
def draw_board(board):
    screen.fill(BLACK)
    # Draw slots
    for c in range(COLS):
        for r in range(ROWS):
            pygame.draw.rect(screen, BLUE,
                             (c * SQUARESIZE, (r + 1) * SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, BLACK,
                               (c * SQUARESIZE + SQUARESIZE // 2,
                                (r + 1) * SQUARESIZE + SQUARESIZE // 2), RADIUS)

    # Draw pieces
    for c in range(COLS):
        for r in range(ROWS):
            if board[r][c] == 1:      # Human (red)
                pygame.draw.circle(screen, RED,
                                   (c * SQUARESIZE + SQUARESIZE // 2,
                                    (r + 1) * SQUARESIZE + SQUARESIZE // 2), RADIUS)
            elif board[r][c] == -1:   # AI (yellow)
                pygame.draw.circle(screen, YELLOW,
                                   (c * SQUARESIZE + SQUARESIZE // 2,
                                    (r + 1) * SQUARESIZE + SQUARESIZE // 2), RADIUS)

    # Hover preview
    if turn == 0 and not game_over and hover_col != -1:
        pygame.draw.circle(screen, RED,
                           (hover_col * SQUARESIZE + SQUARESIZE // 2, SQUARESIZE // 2), RADIUS)

    pygame.display.update()

# ------------------ SHOW MESSAGE ------------------
def show_message(text, color=YELLOW):
    label = font_large.render(text, True, color)
    screen.blit(label, (width // 2 - label.get_width() // 2, 10))
    restart_label = font_small.render("Press R to play again", True, WHITE)
    screen.blit(restart_label, (width // 2 - restart_label.get_width() // 2, 90))
    pygame.display.update()

# ------------------ MAIN GAME LOOP ------------------
draw_board(env.board)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.MOUSEMOTION and turn == 0 and not game_over:
            hover_col = event.pos[0] // SQUARESIZE
            draw_board(env.board)

        if event.type == pygame.MOUSEBUTTONDOWN and turn == 0 and not game_over:
            col = event.pos[0] // SQUARESIZE
            if col in env.valid_actions():
                _, _, done = env.step(col)
                draw_board(env.board)
                if done:
                    winner = env.check_winner()
                    if winner == 1:
                        show_message("You Win!", RED)
                    else:
                        show_message("It's a Draw!", WHITE)
                    game_over = True
                else:
                    turn = 1

        if event.type == pygame.KEYDOWN and event.key == pygame.K_r and game_over:
            env.reset()
            game_over = False
            turn = 0
            hover_col = -1
            draw_board(env.board)

    # AI turn
    if turn == 1 and not game_over:
        state = env.board.flatten().copy()
        with torch.no_grad():
            q_values = model(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
            q_values = q_values[0].cpu()

            # Mask invalid moves
            for c in range(COLS):
                if c not in env.valid_actions():
                    q_values[c] = -float('inf')

            ai_col = torch.argmax(q_values).item()

        pygame.time.wait(600)  # Small delay so you can see the move
        _, _, done = env.step(ai_col)
        draw_board(env.board)

        if done:
            winner = env.check_winner()
            if winner == -1:
                show_message("AI Wins!", YELLOW)
            else:
                show_message("It's a Draw!", WHITE)
            game_over = True

        turn = 0
