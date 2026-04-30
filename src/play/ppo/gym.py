import os
import torch
import numpy as np
from typing import List, Tuple

from ..core.rules import Rules
from ..core.simulator import Simulator
from ..core.agent import Agent
from ..types import Record
from ..kits import ActionKit
from src.types import Trump
from src.kits import CardKit, TrumpKit, ListKit

from .agent import PpoAgent
from .network import PPONetwork
from ..helper_agents.aggresive_player import AggressivePlayer
from ..helper_agents.soft_player import SoftPlayer
from ..helper_agents.random_chooser import RandomChooser


class Gym:
    def __init__(self, model_dir: str = "models", seed: int = 42):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        # Initialize RNG with seed for reproducibility
        self.rng = np.random.default_rng(seed)

        # Initialize Network and Agent
        self.network = PPONetwork().to(self.device)
        self.agent = PpoAgent(self.network, rng=self.rng)

        # Rules
        self.rules = Rules()

        # Predefined opponents
        self.soft_player       = SoftPlayer()
        self.agent_player      = PpoAgent(self.network, rng=self.rng)
        self.randomer_player   = RandomChooser()
        self.aggressive_player = AggressivePlayer()

        print(f"Gym initialized on device: {self.device}")

    def train_phases(
        self,
        opponent_types: List[str],
        num_phases: int,
        games_per_phase: int,
    ):
        self.opponent_types  = opponent_types
        self.num_phases      = num_phases
        self.games_per_phase = games_per_phase

        print("\nStarting Phase-Based Training")
        print(f"Total Phases: {self.num_phases}")
        print(f"Games per Phase: {self.games_per_phase}")
        print(f"Opponent Types: {self.opponent_types}\n")

        phase_rewards = []

        for phase in range(1, self.num_phases + 1):
            self._display_title(f"PHASE {phase}/{self.num_phases}")

            records, scores = self.play_games()

            if records:
                print(f"\nTraining on {len(records)} experiences...")
                metrics = self.agent.learn(records)
                print("✓ Training complete")
                if metrics:
                    print(f"  Loss: {metrics['total_loss']:.4f} "
                          f"(Policy: {metrics['policy_loss']:.4f}, "
                          f"Value: {metrics['value_loss']:.4f}, "
                          f"Entropy: {metrics['entropy']:.4f})")
                    print(f"  Reward vs Loss: Avg Reward={scores:.2f}, Total Loss={metrics['total_loss']:.4f}")
                    
                    # Display per-network metrics
                    print("\n  Per-Network Metrics:")
                    for net_id in range(4):
                        if f'network_{net_id}_policy_loss' in metrics:
                            p_loss = metrics[f'network_{net_id}_policy_loss']
                            v_loss = metrics[f'network_{net_id}_value_loss']
                            samples = metrics[f'network_{net_id}_samples']
                            print(f"    Network {net_id} (table has {net_id} cards): "
                                  f"Policy={p_loss:.4f}, Value={v_loss:.4f}, Samples={samples}")

            # Save model for next phase
            model_path = os.path.join(self.model_dir, "model.pt")
            self.save_model(model_path)

            # Reload agent with latest model
            network = PPONetwork().to(self.device)
            network.load_state_dict(torch.load(model_path, map_location=self.device))
            network.eval()

            self.network      = network
            self.agent        = PpoAgent(network, rng=self.rng)
            self.agent_player = PpoAgent(network, rng=self.rng)

            phase_rewards.append(scores)

    def play_games(self) -> Tuple[List[Record], float]:
        records: List[Record] = []
        total_scores = []

        for _ in range(1, self.games_per_phase + 1):
            # Setup game: shuffle 32 cards (0..31) and deal 8 to each of 4 players
            deck = list(range(32))
            self.rng.shuffle(deck)
            hands = [deck[i * 8:(i + 1) * 8] for i in range(4)]

            # Random trump (int: 0-3 card trump, 4 no-trump, 5 all-trump)
            trump = self._get_random_trump()

            # Canonicalize (transform suits based on player 0's hand strength)
            can_map = ListKit.suit_canonical_map(hands[0])
            if TrumpKit.is_card_trump(trump):
                trump = can_map[TrumpKit.suit(trump)]
            hands = [
                [CardKit.make(can_map[CardKit.suit(c)], CardKit.rank(c)) for c in hand]
                for hand in hands
            ]

            # Create opponents
            opponents = self._select_opponents(self.opponent_types)

            # Assign agents (PPO is player 0)
            agents = [self.agent] + opponents

            # Simulate game
            simulator = Simulator(self.rules, agents, display=False)
            start_player = int(self.rng.integers(0, 4))

            scores, game_records = simulator.simulate((hands, trump, start_player))

            # Compute GAE for card play
            self._compute_play_card_gae(game_records)

            # Collect records for PPO agent (player 0)
            agent_records = [r for r in game_records if r[0] == 0]
            records.extend(agent_records)

            # Track rewards
            total_scores.append(scores[0] - scores[1])

        return records, float(np.mean(total_scores)) if total_scores else 0.0

    def _select_opponents(self, opponent_names: list) -> List[Agent]:
        assert len(opponent_names) >= 3, "At least 3 opponent types must be provided."

        players = {
            'random':     self.randomer_player,
            'aggressive': self.aggressive_player,
            'soft':       self.soft_player,
            'agent':      self.agent_player,
        }

        assert all(opp.lower() in players for opp in opponent_names), \
            f"Invalid opponent types in {opponent_names}. Valid types are: {list(players.keys())}"

        selected = self.rng.choice(opponent_names, size=3, replace=True)
        return [players[opp.lower()] for opp in selected]

    def _display_title(self, title: str):
        print(f"\n{'='*30}{title}{'='*30}")

    def _get_random_trump(self) -> Trump:
        """Generate a random trump int: 0-3 = card trump (suit), 4 = no-trump, 5 = all-trump."""
        return int(self.rng.integers(0, 6))

    def _compute_play_card_gae(self, records: List[Record], gamma: float = 0.99, lam: float = 0.95):
        # Record is a tuple: (player, state, action, reward, log)
        # reward is a tuple: (instant, accrued)
        play_records = [r for r in records if ActionKit.is_play_card(r[2])]
        if not play_records:
            return

        rewards = [0.0] * 9
        for r in play_records:
            player  = r[0]
            state   = r[1]
            accrued = r[3][1]
            rd      = min(state['round'], 7)
            if player in (1, 3):
                rewards[rd] -= accrued / 100.0
            elif player == 2:
                rewards[rd] += accrued / (3 * 100.0)  # reward but not that much
            elif player == 0:
                rewards[rd] += accrued / 100.0

        # Extract only player 0 records
        player_play_records = [r for r in play_records if r[0] == 0]
        if not player_play_records:
            return

        values = [float(r[4]['value']) for r in player_play_records]
        values.append(0.0)

        aligned_rewards = [rewards[min(r[1]['round'], 7)] for r in player_play_records]

        gae = 0.0
        for i in reversed(range(len(player_play_records))):
            delta = aligned_rewards[i] + gamma * values[i + 1] - values[i]
            gae   = delta + gamma * lam * gae

            player_play_records[i][4]['advantage'] = gae
            player_play_records[i][4]['return']    = gae + values[i]

    def save_model(self, path: str):
        torch.save(self.network.state_dict(), path)

    def load_model(self, path: str):
        if os.path.exists(path):
            self.network.load_state_dict(torch.load(path, map_location=self.device))
            self.network.to(self.device)
        else:
            print(f"Warning: Model file not found at {path}")
