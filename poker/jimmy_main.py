import time
from typing import Any, Dict

from stable_baselines3 import PPO

from check_submission import check_submission
from game_mechanics import choose_move_randomly, load_pkl, play_game, save_pkl, PokerEnv

TEAM_NAME = "Team JImmy"  # <---- Enter your team name here!
assert TEAM_NAME != "Team Name", "Please change your TEAM_NAME!"


def train() -> Dict:
    t1 = time.time()

    env = PokerEnv(choose_move_randomly, verbose=False, render=False)
    env.reset()

    model = PPO("MlpPolicy", env, verbose=2)

    model.learn(total_timesteps=100_000)
    t2 = time.time()
    print(t2 - t1)

    model.save("meaty_model")

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
    return model


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
    file = train()
    save_pkl(file, TEAM_NAME)

    check_submission(
        TEAM_NAME
    )  # <---- Make sure I pass! Or your solution will not work in the tournament!!

    my_value_fn = load_pkl(TEAM_NAME)

    # Code below plays a single game against a random
    #  opponent, think about how you might want to adapt this to
    #  test the performance of your algorithm.
    def choose_move_no_value_fn(state: Any) -> int:
        """The arguments in play_game() require functions that only take the state as input.

        This converts choose_move() to that format.
        """
        return choose_move(state, my_value_fn)

    play_game(
        your_choose_move=choose_move_no_value_fn,
        opponent_choose_move=choose_move_randomly,
        game_speed_multiplier=1,
        render=True,
        verbose=False,
    )
