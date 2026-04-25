from typing import List
from src.types import Card
from src.kits import CardKit, ListKit
from ..kits import StateKit, ActionKit
from ..types import Record, Game, Scores
from .rules import Rules
from .agent import Agent


class Simulator:
    def __init__(self, rules: Rules, agents: List[Agent], display: bool = False):
        self.rules   = rules
        self.agents  = agents
        self.display = display

    # Returns Scores and Records
    def simulate(self, game: Game) -> tuple[Scores, List[Record]]:
        hands, trump, start_player = game[0], game[1], game[2]

        # Initialize game state
        self.scores = [0, 0]
        self.trump  = trump

        # Per-player state. StateKit wraps a dict and proxies attribute access.
        # Trump is stored on the state so rules can read it.
        self.states: List[StateKit] = []
        for i in range(4):
            s = StateKit.make(hands[i])
            s['trump'] = trump
            self.states.append(StateKit(s))

        # Records
        records: List[Record] = []

        # 32 cards total
        cards_moves = 0
        next_player = start_player

        # Player → index of their last play_card record (for lazy accrued reward update)
        player_last_card_played_record_index: dict[int, int] = {}

        self._display_line()

        while cards_moves < 32:
            current_player = next_player
            current_agent  = self.agents[current_player]
            current_state  = self.states[current_player]

            # Get valid actions
            actions = self.rules.actions(current_state)

            # Agent chooses action
            action, log = current_agent.choose_action(current_state, actions)

            # Recompute the current table snapshot after observers updated state
            current_table = current_state.table.copy()

            # ----------------------------#
            # ---- Other actions here ----#
            # ----------------------------#
            # .....

            # ----------------------------#
            # ----- CARD PLAY ACTION -----#
            # ----------------------------#
            if ActionKit.is_play_card(action):
                card = ActionKit.value(action)
                current_table.append(card)

                self._display_table(current_table)

                cards_moves += 1

                # Winner card and its index on the table
                _, winner_table_idx = ListKit.winner(current_table, trump)

                # Table points
                table_points = ListKit.value(current_table, trump)

                # Winner player index (absolute).
                winner_player_index = (current_player - (len(current_table) - 1) + winner_table_idx) % 4

                # Instant reward for the player
                instant_reward = table_points if winner_player_index == current_player else 0

                # Append record. Reward is (instant, accrued).
                player_last_card_played_record_index[current_player] = len(records)
                records.append((current_player, current_state, action, (instant_reward, 0), log))

                # Continue to next player
                next_player = (current_player + 1) % 4

                # END OF TRICK PROCESSING
                if cards_moves % 4 == 0:
                    # Lazy accrued reward update for the trick winner's last play.
                    # Records are tuples → rebuild with updated reward.
                    idx = player_last_card_played_record_index[winner_player_index]
                    p, st, ac, (inst, _), lg = records[idx]
                    records[idx] = (p, st, ac, (inst, table_points), lg)

                    # Calculate points
                    self.scores[winner_player_index % 2] += table_points

                    # Set winner as next player
                    next_player = winner_player_index

                    # Display Table after action
                    self._display_line()

            # Update all states
            for i in range(4):
                self.states[i].observe(self._relative_player(i, current_player), action)

        self._display_summary()

        return self.scores, records

    def _relative_player(self, current_player: int, relative_to: int) -> int:
        """Get player index relative to another player"""
        return (current_player - relative_to) % 4

    def _display_table(self, current_table: List[Card]):
        if not self.display:
            return
        table_cards = [CardKit.str(c) for c in current_table]
        while len(table_cards) < 4:
            table_cards.append(" . ")
        print(f"Table: {' '.join(table_cards)}")

    def _display_summary(self):
        if not self.display:
            return
        round_gain = self.scores[0]
        round_loss = self.scores[1]
        win_or_lost = '[0]' if round_gain > round_loss else '[1]'

        self._display_line()
        print(f"Total: Winner is {win_or_lost}, Total ({round_gain}, {round_loss})")
        self._display_line()

    def _display_line(self):
        if not self.display:
            return
        print("")
        print("-----------------------------------------")
        print("")
