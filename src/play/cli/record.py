import random
import sys
import os
import pickle
import argparse

# Add the project root to the python path
sys.path.append(os.getcwd())

from src.play.cli.utils.main import transform_canonical
from src.kits import ListKit
from src.play.core.simulator import Simulator
from src.play.core.rules import Rules
from src.play.helper_agents.human import Human


def main():
    parser = argparse.ArgumentParser(description="Record Belote Game")
    parser.add_argument("--output", type=str, default="records/record-001.pkl", help="Path to save the record file")
    args = parser.parse_args()

    # Randomly choose trump (0-3 = card trump, 4 = no-trump, 5 = all-trump)
    trump = random.randint(0, 3)

    # Hands: shuffle 32 cards into 4 hands of 8, then sort each
    deck = list(range(32))
    random.shuffle(deck)
    hands = [ListKit.sort(deck[i * 8:(i + 1) * 8], trump) for i in range(4)]

    # All Human
    agents = [Human() for _ in range(4)]

    # Initialize Simulator
    rules = Rules()
    simulator = Simulator(rules, agents, display=True)

    # Transform to canonical form
    trump, hands = transform_canonical(trump, hands)

    print("")
    print("Starting Belote Game Recording...")
    print(f"Trump is {trump}")
    print("You will play for all 4 players.")

    # Run simulation
    start_player = random.randint(0, 3)
    scores, records = simulator.simulate((hands, trump, start_player))

    print("\nGame Over!")
    print(f"Total Scores: {scores}")

    # Save the result (hands, trump, start_player, records)
    save_path = args.output
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump({
            "hands":        hands,
            "trump":        trump,
            "start_player": start_player,
            "records":      records,
        }, f)
    print(f"Game recorded to {save_path}")


if __name__ == "__main__":
    main()

