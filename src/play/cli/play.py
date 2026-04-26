import random
import sys
import os
import torch
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.play.cli.utils.transformer import transform_canonical
from src.kits import ListKit, TrumpKit
from src.play.core.simulator import Simulator
from src.play.core.rules import Rules
from src.play.helper_agents.human import Human
from src.play.ppo.agent import PpoAgent
from src.play.ppo.network import PPONetwork


def main():
    parser = argparse.ArgumentParser(description="Play Belote against PPO Agent")
    parser.add_argument("--model", type=str, default="models/model.pt", help="Path to the trained model file")
    args = parser.parse_args()

    # Randomly choose trump (0-3 = card trump, 4 = no-trump, 5 = all-trump)
    trump = random.randint(0, 3)

    # Hands: shuffle 32 cards into 4 hands of 8, then sort each
    deck = list(range(32))
    random.shuffle(deck)
    hands = [ListKit.sort(deck[i * 8:(i + 1) * 8], trump) for i in range(4)]

    # Initialize agents
    network = PPONetwork()
    network.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))
    print(f"Loaded model from {args.model}")

    agents = [Human()] + [PpoAgent(network) for _ in range(3)]

    # Initialize Simulator
    rules = Rules()
    simulator = Simulator(rules, agents, display=True)

    # Transform to canonical form
    trump, hands = transform_canonical(trump, hands)

    print("")
    print("Starting Belote Game Simulation...")
    print(f"{TrumpKit.str(trump)} | You are Player 0.")

    # Run simulation
    start_player = random.randint(0, 3)
    scores, _ = simulator.simulate((hands, trump, start_player))

    print("\nGame Over!")
    print(f"Total Scores: {scores}")


if __name__ == "__main__":
    main()

