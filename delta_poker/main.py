import numpy as np

from check_submission import check_submission
from game_mechanics import (
    ChooseMoveCheckpoint,
    PokerEnv,
    checkpoint_model,
    choose_move_randomly,
    human_player,
    load_network,
    play_poker,
    save_network,
)
from torch import nn

TEAM_NAME = "Team Name"  # <---- Enter your team name here!
assert TEAM_NAME != "Team Name", "Please change your TEAM_NAME!"


def train() -> nn.Module:
    """
    TODO: Write this function to train your network.

    Returns:
        pytorch network
    """

    # Below is an example of how to checkpoint your model and
    # the train against these checkpoints
    env = PokerEnv(choose_move_randomly)
    model = ...
    # Train model in env .........
    checkpoint_model(model, "checkpoint1")
    env = PokerEnv(ChooseMoveCheckpoint("checkpoint1", choose_move).choose_move)
    # Train model in env .........

    raise NotImplementedError("You need to implement this function!")


def choose_move(state: np.ndarray, legal_moves: np.ndarray, neural_network: nn.Module) -> int:
    """Called during competitive play. It acts greedily given current state of the board and your
    network. It returns a single move to play.

    Args:
         state: The state of poker game. shape = (72,)
         legal_moves: Legal actions on this turn. Subset of {0, 1, 2, 3}
         neural_network: Your pytorch network from train()

    Returns:
        action: Single value drawn from legal_moves
    """
    raise NotImplementedError("You need to implement this function!")


if __name__ == "__main__":

    # Example workflow, feel free to edit this! ###
    neural_network = train()
    save_network(neural_network, TEAM_NAME)

    check_submission(
        TEAM_NAME
    )  # <---- Make sure I pass! Or your solution will not work in the tournament!!

    neural_network = load_network(TEAM_NAME)

    # Code below plays a single game against a random
    #  opponent, think about how you might want to adapt this to
    #  test the performance of your algorithm.
    def choose_move_no_network(state: np.ndarray, legal_moves: np.ndarray) -> int:
        """The arguments in play_poker() require functions that only take the state as input.

        This converts choose_move() to that format.
        """
        return choose_move(state, legal_moves, neural_network)

    # Challenge your bot to a game of poker!
    play_poker(
        your_choose_move=human_player,
        opponent_choose_move=choose_move_no_network,
        game_speed_multiplier=1000,
        render=True,
        verbose=True,
    )
