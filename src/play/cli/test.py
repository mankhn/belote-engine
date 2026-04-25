import os
import torch
import pickle
import argparse

from src.play.core.rules import Rules
from src.play.core.simulator import Simulator
from src.play.ppo.network import PPONetwork
from src.play.helper_agents.ppo_tester import PpoTester


def main():
    parser = argparse.ArgumentParser(description="Test Belote PPO Agent against recorded game")
    parser.add_argument("--model", type=str, default="models/model.pt", help="Path to the trained model file")
    parser.add_argument("--record", type=str, default="records/record-001.pkl", help="Path to the record file")
    parser.add_argument("--times", type=int, default=1, help="Number of times to play the game")
    args = parser.parse_args()

    # Load the recorded game
    load_path = args.record
    if not os.path.exists(load_path):
        print(f"Error: {load_path} not found. Please run record.py first.")
        return

    print(f"Loading recorded game from {load_path}...")
    with open(load_path, "rb") as f:
        result = pickle.load(f)

    records      = result["records"]
    hands        = result["hands"]
    trump        = result["trump"]
    start_player = result["start_player"]

    if not records:
        print("No records found in the loaded game.")
        return

    # Initialize PPO Agent
    print("Loading PPO Agent...")
    network = PPONetwork()
    model_path = args.model

    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Please run training first.")
        return

    network.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    print(f"Loaded model from {model_path}")

    # Initialize Rules
    rules = Rules()

    # Reset global statistics
    PpoTester.total_moves = 0
    PpoTester.total_matches = 0

    print(f"Starting {args.times} comparison simulation(s)...")

    # Run simulations N times
    for _ in range(args.times):
        # Reset cursor for each game
        PpoTester.cursor = 0

        # Create new agents for this game
        agents = [PpoTester(network, records=records) for _ in range(4)]

        # Initialize Simulator
        simulator = Simulator(rules, agents, display=False)

        # Run simulation using hands and trump from the recorded game
        simulator.simulate((hands, trump, start_player))

    # Print combined stats
    if PpoTester.total_moves > 0:
        accuracy = (PpoTester.total_matches / PpoTester.total_moves) * 100
        print("\nComparison Complete.")
        print(f"Total Games: {args.times}")
        print(f"Total Moves: {PpoTester.total_moves}")
        print(f"Matches: {PpoTester.total_matches}")
        print(f"Accuracy: {accuracy:.2f}%")
    else:
        print("\nNo moves were simulated.")


if __name__ == "__main__":
    main()

