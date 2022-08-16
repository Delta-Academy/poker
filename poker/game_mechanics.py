import random
from typing import Any, Callable, Dict, Tuple
from pettingzoo.classic import texas_holdem_v4

import numpy as np

from env_wrapper import DeltaEnv, wrap_env


def save_pkl(file: Any, team_name: str):
    """Save a user PKL."""
    pass


def load_pkl(team_name: str):
    """Load a user PKL."""
    pass


def choose_move_randomly(observation, legal_moves):
    # Maybe need this not sure
    # if len(legal_moves) == 0:
    #     return None
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
