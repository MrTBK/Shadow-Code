import chess
import torch
import torch.nn as nn
import torch.optim as optim
import random
import sys
import os
import matplotlib.pyplot as plt

STATE_DIM  = 12 * 8 * 8 
ACTION_DIM = 4672       

class DQN(nn.Module):
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        return self.net(x)

_MOVE_LIST   = [m.uci() for m in chess.Board().legal_moves] * 20
_MOVE_TO_IDX = {uci: idx for idx, uci in enumerate(_MOVE_LIST)}

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    arr = torch.zeros((12, 8, 8), dtype=torch.float32)
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            plane = (p.piece_type - 1) + (0 if p.color else 6)
            r, c = divmod(sq, 8)
            arr[plane, 7-r, c] = 1.0
    return arr.view(1, -1)

def move_to_idx(move: chess.Move) -> int:
    return _MOVE_TO_IDX.get(move.uci(), 0)

class ChessEnv:
    def __init__(self): self.board = chess.Board()
    def reset(self): self.board.reset(); return self.board
    def legal_moves(self): return list(self.board.legal_moves)
    def step(self, move):
        self.board.push(move)
        done = self.board.is_game_over()
        res = self.board.result() if done else None
        reward = 1 if res == '1-0' else -1 if res == '0-1' else 0
        return self.board, reward, done

import os

def load_dqn(path: str = "dqn_checkpoint.pt") -> DQN:

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file '{path}' not found.")
    ckpt = torch.load(path, map_location="cpu")
    model = DQN()
    if isinstance(ckpt, dict) and 'model' in ckpt:
        state = ckpt['model']
    else:
        state = ckpt
    model.load_state_dict(state)
    model.eval()
    return model

if __name__ == "__main__":
    checkpoint = "dqn_checkpoint.pt"
    policy_net = DQN()
    target_net = DQN()
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    episode = 0
    rewards = []
    if os.path.exists(checkpoint):
        ckpt = torch.load(checkpoint, map_location="cpu")
        state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        policy_net.load_state_dict(state)
        target_net.load_state_dict(state)
        episode = ckpt.get('episode', 0) if isinstance(ckpt, dict) else 0
        rewards = ckpt.get('rewards', []) if isinstance(ckpt, dict) else []
        print(f"Resumed at episode {episode}")
    else:
        target_net.load_state_dict(policy_net.state_dict())
        print("Starting fresh training")

    env = ChessEnv()
    plt.ion(); fig, ax = plt.subplots(figsize=(8,5))
    ax.set_title("DQN Training Progress")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.grid(True, linestyle='--', linewidth=0.5)
    line, = ax.plot([], [], 'o-', label="Reward")
    ax.legend()

    try:
        while True:
            episode += 1
            board = env.reset()
            done = False; total = 0
            while not done:
                mv = random.choice(env.legal_moves())
                board, r, done = env.step(mv)
                total += r
            rewards.append(total)
            x = list(range(1, episode+1))
            line.set_data(x, rewards)
            ax.set_xlim(1, max(10, episode))
            ax.set_ylim(min(rewards)-1, max(rewards)+1)
            fig.canvas.draw(); fig.canvas.flush_events()
            if episode % 10 == 0:
                print(f"Episode {episode} | Reward {total}")
                torch.save({'episode':episode,'model':policy_net.state_dict(),'rewards':rewards}, checkpoint)
    except KeyboardInterrupt:
        print(f"Stopped at episode {episode}")
        torch.save({'episode':episode,'model':policy_net.state_dict(),'rewards':rewards}, checkpoint)
        sys.exit()
