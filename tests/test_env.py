import pytest

from delta_poker.env_wrapper import DeltaEnv
from pettingzoo.classic import texas_holdem_no_limit_v6


def raiser(state, legal_moves):
    """Always raises."""
    return 3


def test_fold_looses():
    env = DeltaEnv(texas_holdem_no_limit_v6.env(), raiser, verbose=False, render=False)
    env.reset()
    # Raise
    _, reward, done, _ = env.step(3)
    assert reward == 0

    # Fold
    _, reward, done, _ = env.step(0)
    assert reward < 0
