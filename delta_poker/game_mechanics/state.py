from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class State:
    # TODO: Write docstring
    @staticmethod
    def from_dict(state_dict: Dict):
        assert (
            len(state_dict["all_chips"]) == 2
        ), "State class doesn't support games of more than 2 players!"
        player = state_dict["all_chips"].index(state_dict["my_chips"])
        opponent = 1 - player
        return State(
            hand=state_dict["hand"],
            public_cards=state_dict["public_cards"],
            player_chips=state_dict["my_chips"],
            opponent_chips=state_dict["all_chips"][opponent],
            player_chips_remaining=state_dict["stakes"][player],
            opponent_chips_remaining=state_dict["stakes"][opponent],
            stage=state_dict["stage"].value,
            legal_actions=[action.value for action in state_dict["legal_actions"]],
        )

    hand: List[str]
    public_cards: List[str]
    player_chips: int
    opponent_chips: int
    player_chips_remaining: int
    opponent_chips_remaining: int
    stage: int
    legal_actions: List[int]


def to_basic_nn_input(state: State) -> torch.Tensor:
    nn_input = torch.zeros(55)
    suits = ["C", "D", "H", "S"]
    ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
    for card in state.hand:
        nn_input[13 * suits.index(card[0]) + ranks.index(card[1])] = 1
    for card in state.public_cards:
        nn_input[13 * suits.index(card[0]) + ranks.index(card[1])] = -1
    nn_input[52] = state.player_chips / 100
    nn_input[53] = state.opponent_chips / 100
    nn_input[54] = (state.player_chips_remaining / 100) - 1
    return nn_input
