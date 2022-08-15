import pytest

from poker.texas_holdem import DeltaEnv
from pettingzoo.classic import texas_holdem_v4


def raiser(state, legal_moves):
    """Always raises"""
    return 1


def test_fold_looses():
    env = DeltaEnv(texas_holdem_v4.env(), raiser, verbose=False, render=False)
    env.reset()
    # Raise
    _, reward, done, _ = env.step(1)
    assert reward == 0
    # Fold
    _, reward, done, _ = env.step(2)
    assert reward < 0


def test_illegal_move():
    env = DeltaEnv(texas_holdem_v4.env(), raiser, verbose=False, render=False)
    env.reset()
    # Raise
    _, reward, done, _ = env.step(1)
    assert reward == 0
    # Cant call after an opponent raise
    _, reward, done, _ = env.step(3)
    assert done
    assert reward == -10
