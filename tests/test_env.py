import pytest
from pettingzoo.classic import texas_holdem_v4

from delta_poker.env_wrapper import DeltaEnv


def raiser(state, legal_moves):
    """Always raises."""
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
    assert done


def test_illegal_move():
    env = DeltaEnv(texas_holdem_v4.env(), raiser, verbose=False, render=False)
    env.reset()
    _, reward, done, _ = env.step(1)
    assert reward == 0
    assert done == False
    with pytest.raises(AssertionError):
        # Cant call after an opponent raise
        _, _, _, _ = env.step(3)
