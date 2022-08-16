import random
from typing import Any, Callable, Dict, Tuple

import numpy as np
import torch
from pettingzoo.classic import texas_holdem_v4
from stable_baselines3 import PPO
from torch import nn

from env_wrapper import DeltaEnv


def checkpoint_model(model: nn.Module, checkpoint_name: str):
    # torch.save(model, checkpoint_name)
    model.save(checkpoint_name)


def load_checkpoint(checkpoint_name: str):
    return PPO.load(checkpoint_name)
    # return torch.load(checkpoint_name)


def load_pkl(team_name: str):
    """Load a user PKL."""
    pass


def choose_move_randomly(observation, legal_moves):
    return random.choice(legal_moves)


def choose_move_rules(observation, legal_moves):
    cards = observation[:52]
    if sum(cards != 0) == 2:
        # Don't fold before the flop
        legal_moves = legal_moves[legal_moves != 2]
        return random.choice(legal_moves)

    suits = cards.reshape(4, 13)
    # Pair, 3 etc
    has_matches = np.any(np.sum(suits, 1) > 1)
    has_aces = np.any(cards[[0, 13, 26, 39]])
    if has_matches or has_aces:
        legal_moves = legal_moves[legal_moves != 2]
        return random.choice(legal_moves)
    else:
        return 2


def choose_move_never_folds(observation, legal_moves):
    legal_moves = legal_moves[legal_moves != 2]
    return random.choice(legal_moves)


def play_game(
    your_choose_move: Callable,
    opponent_choose_move: Callable,
    game_speed_multiplier=1,
    render=True,
    verbose=False,
):
    pass


def PokerEnv(
    opponent_choose_move: Callable[[np.ndarray, np.ndarray], int], verbose: bool, render: bool
):
    return DeltaEnv(texas_holdem_v4.env(), opponent_choose_move, verbose, render)
