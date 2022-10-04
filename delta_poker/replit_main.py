import random
from typing import Callable

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from torch import nn

from game_mechanics import HERE, PokerEnv, choose_move_randomly, human_player, wait_for_click

NEURAL_NETWORK = MaskablePPO.load(
    "/Users/jamesrowland/Code/poker/delta_poker/nolimit_checkpoint1.zip"
)


def bot_choose_move(state: np.ndarray, legal_moves: np.ndarray) -> int:
    """Called during competitive play. It acts greedily given current state of the board and your
    network. It returns a single move to play.

    Args:
         state: The state of poker game. shape = (72,)
         legal_moves: Legal actions on this turn. Subset of {0, 1, 2, 3}
         neural_network: Your pytorch network from train()

    Returns:
        action: Single value drawn from legal_moves
    """

    state = np.abs(state)
    mask = np.isin(np.arange(5), legal_moves)
    action, _ = NEURAL_NETWORK.predict(state, deterministic=False, action_masks=mask)
    return action


def play_poker(
    choose_move: Callable,
    opponent_choose_move: Callable,
    game_speed_multiplier=1,
    render=True,
    verbose=False,
) -> None:

    env = PokerEnv(
        opponent_choose_move,
        verbose=verbose,
        render=render,
        game_speed_multiplier=game_speed_multiplier,
    )

    def mask_fn(env):
        return env.last()[0]["action_mask"]

    env = ActionMasker(env, mask_fn)

    while True:

        observation, reward, done, info = env.reset()

        # observation = env.reset()
        # done = False
        # info = {"legal_moves":  [0, 1, 2, 3, 4]}

        while not done:
            action = choose_move(observation, info["legal_moves"])
            observation, reward, done, info = env.step(action)

        wait_for_click()


if __name__ == "__main__":

    play_poker(
        choose_move=human_player,
        opponent_choose_move=bot_choose_move,
        game_speed_multiplier=100,
        render=True,
        verbose=True,
    )
