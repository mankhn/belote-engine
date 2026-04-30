import torch
import torch.nn as nn
from ..types import State


class Network0Cards(nn.Module):
    """Network for playing when table has 0 cards (leading)"""
    
    def __init__(self, state_dim=256, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.card_head = nn.Linear(hidden_dim, 32)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
    
    def forward(self, state_features):
        features = self.network(state_features)
        return {
            'card_policy': self.card_head(features),
            'value': self.value_head(features)
        }


class Network1Card(nn.Module):
    """Network for playing when table has 1 card"""
    
    def __init__(self, shared_card_embedding, state_dim=256, card_emb_dim=16, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.card_embedding = shared_card_embedding
        
        input_dim = state_dim + card_emb_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.card_head = nn.Linear(hidden_dim, 32)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
    
    def forward(self, state_features, table_cards):
        # table_cards: [B, 4], but only first card is valid
        card_emb = self.card_embedding(table_cards[:, 0])  # [B, card_emb_dim]
        combined = torch.cat([state_features, card_emb], dim=-1)
        features = self.network(combined)
        return {
            'card_policy': self.card_head(features),
            'value': self.value_head(features)
        }


class Network2Cards(nn.Module):
    """Network for playing when table has 2 cards"""
    
    def __init__(self, shared_card_embedding, state_dim=256, card_emb_dim=16, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.card_embedding = shared_card_embedding
        
        input_dim = state_dim + card_emb_dim * 2
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.card_head = nn.Linear(hidden_dim, 32)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
    
    def forward(self, state_features, table_cards):
        # table_cards: [B, 4], first 2 cards are valid
        card_emb = self.card_embedding(table_cards[:, :2])  # [B, 2, card_emb_dim]
        card_emb_flat = card_emb.reshape(card_emb.size(0), -1)  # [B, 2*card_emb_dim]
        combined = torch.cat([state_features, card_emb_flat], dim=-1)
        features = self.network(combined)
        return {
            'card_policy': self.card_head(features),
            'value': self.value_head(features)
        }


class Network3Cards(nn.Module):
    """Network for playing when table has 3 cards"""
    
    def __init__(self, shared_card_embedding, state_dim=256, card_emb_dim=16, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.card_embedding = shared_card_embedding
        
        input_dim = state_dim + card_emb_dim * 3
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.card_head = nn.Linear(hidden_dim, 32)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
    
    def forward(self, state_features, table_cards):
        # table_cards: [B, 4], first 3 cards are valid
        card_emb = self.card_embedding(table_cards[:, :3])  # [B, 3, card_emb_dim]
        card_emb_flat = card_emb.reshape(card_emb.size(0), -1)  # [B, 3*card_emb_dim]
        combined = torch.cat([state_features, card_emb_flat], dim=-1)
        features = self.network(combined)
        return {
            'card_policy': self.card_head(features),
            'value': self.value_head(features)
        }


class PPONetwork(nn.Module):
    """
    4 Completely Separate Networks for Different Table States
    
    Each network is independently trained and has no shared parameters:
    - Network0Cards: Hand + Probability → Output (0 table cards)
    - Network1Card:  Hand + Probability + 1 Table Card → Output (1 table card)
    - Network2Cards: Hand + Probability + 2 Table Cards → Output (2 table cards)
    - Network3Cards: Hand + Probability + 3 Table Cards → Output (3 table cards)
    """

    def __init__(self, state_dim=256, hidden_dim=256, card_emb_dim=16, dropout=0.1):
        super().__init__()

        self.num_cards = 32
        
        # Shared card embedding across all networks (reduces parameters by ~75% for embeddings)
        self.shared_card_embedding = nn.Embedding(37, card_emb_dim, padding_idx=36)
        
        # Four completely separate networks (but sharing the card embedding)
        self.network_0cards = Network0Cards(state_dim, hidden_dim, dropout)
        self.network_1card = Network1Card(self.shared_card_embedding, state_dim, card_emb_dim, hidden_dim, dropout)
        self.network_2cards = Network2Cards(self.shared_card_embedding, state_dim, card_emb_dim, hidden_dim, dropout)
        self.network_3cards = Network3Cards(self.shared_card_embedding, state_dim, card_emb_dim, hidden_dim, dropout)

    def get_network_parameters(self, network_id: int):
        """Get parameters for a specific network (0-3)"""
        networks = [self.network_0cards, self.network_1card, self.network_2cards, self.network_3cards]
        return networks[network_id].parameters()
    
    def get_shared_embedding_parameters(self):
        """Get parameters for the shared card embedding"""
        return [self.shared_card_embedding.weight]

    def forward(self, state: State):
        """
        Forward pass - routes to appropriate network based on table card count.
        
        Args:
            state: State object with probabilities [B, 128], history [B, 128] and tables [B, 4]
        
        Returns:
            dict with card_policy and value
        """
        probabilities = state['probabilities']  # [B, 128]
        history = state['history']               # [B, 128]
        tables = state['tables']                 # [B, 4]

        B = probabilities.size(0)
        device = probabilities.device

        # Combine state features (Hand + Probability)
        state_features = torch.cat([probabilities, history], dim=-1)  # [B, 256]
        
        # Prepare table cards for embedding (replace -1 with padding index 36)
        table_cards = tables.clone()
        table_cards[table_cards < 0] = 36
        
        # Count valid cards on table
        valid_mask = (tables >= 0) & (tables < 32)
        counts = valid_mask.sum(dim=1)  # [B]
        counts = torch.clamp(counts, 0, 3)  # Ensure in range [0, 3]
        
        # Initialize outputs
        card_logits = torch.zeros(B, self.num_cards, device=device)
        values = torch.zeros(B, 1, device=device)
        
        # Route to appropriate network based on table card count
        for num_cards in range(4):
            mask = (counts == num_cards)
            if not mask.any():
                continue
            
            if num_cards == 0:
                outputs = self.network_0cards(state_features[mask])
            elif num_cards == 1:
                outputs = self.network_1card(state_features[mask], table_cards[mask])
            elif num_cards == 2:
                outputs = self.network_2cards(state_features[mask], table_cards[mask])
            else:  # num_cards == 3
                outputs = self.network_3cards(state_features[mask], table_cards[mask])
            
            card_logits[mask] = outputs['card_policy']
            values[mask] = outputs['value']
        
        # Apply action masking
        if state.get('legal_actions') is not None:
            card_logits = card_logits + state['legal_actions']
        
        return {
            "card_policy": card_logits,  # [B, 32]
            "value": values,  # [B, 1]
            "table_counts": counts,  # [B] - which network was used for each sample
        }
