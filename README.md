# Shadow-Code

## Introduction
Shadow-Code is a chess game that allows both local play against a DRL (Deep Reinforcement Learning) agent and online multiplayer. The game is designed to provide an engaging experience with real-time multiplayer capabilities, a DRL agent for AI play, and an easy-to-use interface.

## Features
- **Local play vs DRL**: Play against a trained AI powered by Deep Reinforcement Learning (DRL).
- **Online multiplayer**: Play online with other players in real-time using socket-based communication.
- **Piece movement & promotion**: Pieces move automatically when clicked, and pawn promotion is supported with a selection interface.
- **Sounds & UI**: Sound effects for move, capture, and promotion, along with a graphical interface rendered using Pygame.

## Installation

### Clone the repository
```bash
git clone https://github.com/MrTBK/Shadow-Code.git
cd Shadow-Code
```
### Installing Dependencies
Use the following command to install all the required libraries:
```bash
pip install -r requirements.txt
```
### Running the Game
```bash
python game.py
```
### Online Multiplayer Mode

To start a multiplayer game, you can host or join a game:

**Hosting a game:**

In the main menu, select the "Host" option.

The game will open a socket and wait for a connection from another player.

**Joining a game:**

In the main menu, select the "Join" option.

Enter the IP address of the host and join the game.
