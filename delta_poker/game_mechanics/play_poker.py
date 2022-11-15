from typing import Callable

from delta_poker.game_mechanics import PokerEnv
from delta_poker.game_mechanics.render import human_player


def play_poker(
    your_choose_move: Callable,
    opponent_choose_move: Callable,
    game_speed_multiplier: float = 1,
    render: bool = True,
    verbose: bool = False,
) -> None:

    assert (
        opponent_choose_move != human_player
    ), "Set your_choose_move to human_player not opponent_choose_move"

    env = PokerEnv(
        opponent_choose_move,
        verbose=verbose,
        render=render,
        game_speed_multiplier=game_speed_multiplier,
    )

    observation, reward, done, info = env.reset()
    while not done:
        action = your_choose_move(observation)
        observation, reward, done, info = env.step(action)
        print("reward", reward)
        print("\n")
