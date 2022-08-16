import time
from typing import Any, Dict

import numpy as np
from matplotlib import pyplot as plt
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean

from check_submission import check_submission
from game_mechanics import (
    PokerEnv,
    choose_move_never_folds,
    choose_move_randomly,
    choose_move_rules,
    load_checkpoint,
    play_game,
)

TEAM_NAME = "Team Jimmy"  # <---- Enter your team name here!
assert TEAM_NAME != "Team Name", "Please change your TEAM_NAME!"


class ChooseMoveCheckpoint:
    def __init__(self, checkpoint_name: str):
        self.neural_network = load_checkpoint(checkpoint_name)

    def choose_move(self, state, legal_moves):
        neural_network = self.neural_network
        action, _states = neural_network.predict(state, deterministic=True)
        return action


def test_model(model):
    test_env = PokerEnv(choose_move_randomly, verbose=False, render=False)
    n_test_games = 100
    rewards = []
    for _ in range(n_test_games):
        obs = test_env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
        rewards.append(reward)
    print(f"Performance: {np.mean(rewards)}")
    1 / 0


class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.rewards = []

    def _on_step(self) -> None:
        """Called every step()"""
        self.rewards.append(safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer]))


def mask_fn(env):
    return env.last()[0]["action_mask"]


def train() -> Dict:
    #############
    # Play against hard-coded opponent

    env = PokerEnv(choose_move_rules, verbose=True, render=False)

    env = ActionMasker(env, mask_fn)
    env.reset()

    model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=2, ent_coef=0.01)
    env.reset()
    # model = PPO.load("checkpoint1")
    # model.set_env(env)

    callback = CustomCallback()
    t1 = time.time()
    model.learn(total_timesteps=300_000, callback=callback)
    model.save("checkpoint1")
    t2 = time.time()
    print(t2 - t1)
    print("reached checkpoint \n\n\n\n\n")
    plt.plot(callback.rewards)
    plt.show()
    test_model(model)

    1 / 0
    ################
    # Play checkpointed self

    choose_move_checkpoint = ChooseMoveCheckpoint("checkpoint1").choose_move

    env = PokerEnv(choose_move_checkpoint, verbose=True, render=False)
    env.reset()

    model.set_env(env)

    model.learn(total_timesteps=600_000)

    model.save("meaty_model")
    1 / 0
    return model


def test():
    choose_move_checkpoint = ChooseMoveCheckpoint(
        "/Users/jamesrowland/Code/poker/poker/checkpoint1.zip"
    ).choose_move
    # test_env = PokerEnv(choose_move_checkpoint, verbose=True, render=True)
    test_env = PokerEnv(choose_move_randomly, verbose=True, render=True)
    test_env = ActionMasker(test_env, mask_fn)

    model = MaskablePPO.load("/Users/jamesrowland/Code/poker/poker/checkpoint1.zip")

    n_test_games = 100
    rewards = []

    for _ in range(n_test_games):
        obs = test_env.reset()
        done = False
        while not done:
            action, _states = model.predict(
                obs, deterministic=False, action_masks=mask_fn(test_env)
            )
            obs, reward, done, info = test_env.step(action)

        print(f"Game over! Reward {reward}")
        print("\n\n\n\n")

        rewards.append(reward)

    print(f"Performance: {np.mean(rewards)}")


def choose_move(state: Any, user_file: Any, verbose: bool = False) -> int:
    """Called during competitive play. It acts greedily given current state of the board and value
    function dictionary. It returns a single move to play.

    Args:
        state:

    Returns:
    """
    raise NotImplementedError("You need to implement this function!")


if __name__ == "__main__":

    ## Example workflow, feel free to edit this! ###
    # file = train()
    test()

    # save_pkl(file, TEAM_NAME)

    # check_submission(
    #     TEAM_NAME
    # )  # <---- Make sure I pass! Or your solution will not work in the tournament!!

    # my_value_fn = load_pkl(TEAM_NAME)

    # # Code below plays a single game against a random
    # #  opponent, think about how you might want to adapt this to
    # #  test the performance of your algorithm.
    # def choose_move_no_value_fn(state: Any) -> int:
    #     """The arguments in play_game() require functions that only take the state as input.

    #     This converts choose_move() to that format.
    #     """
    #     return choose_move(state, my_value_fn)

    # play_game(
    #     your_choose_move=choose_move_no_value_fn,
    #     opponent_choose_move=choose_move_randomly,
    #     game_speed_multiplier=1,
    #     render=True,
    #     verbose=False,
    # )
