import numpy as np
from torch import nn

from check_submission import check_submission
from game_mechanics import (
    ChooseMoveCheckpoint,
    PokerEnv,
    State,
    checkpoint_model,
    choose_move_randomly,
    human_player,
    load_network,
    play_poker,
    save_network,
    to_basic_nn_input,
)

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

    # Create an env where the opponent is a random player
    env = PokerEnv(choose_move_randomly)
    model = ...
    # Train model in env .........

    # Save the model after the first round of training
    checkpoint_model(model, "checkpoint1.pt")

    # Create a new env, where the opponent is the checkpointed model
    env = PokerEnv(ChooseMoveCheckpoint("checkpoint1.pt", choose_move))

    # Train a new model against the checkpoint...

    raise NotImplementedError("You need to implement this function!")


def choose_move(state: State, neural_network: nn.Module) -> int:
    """Called during competitive play. It returns a single move to play.

    Args:
         state: The state of poker game
         neural_network: Your pytorch network from train()

    Returns:
        action: Single value drawn from legal_actions
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
    #  opponent, think about how you might want to adapt
    #  this to test the performance of your algorithm.
    def choose_move_no_network(state: State) -> int:
        """The arguments in play_poker() require functions that only take the state as input.

        This converts choose_move() to that format.
        """
        return choose_move(state, neural_network)

    # Challenge your bot to a game of poker!
    play_poker(
        your_choose_move=human_player,
        opponent_choose_move=choose_move_no_network,
        game_speed_multiplier=0.5,
        render=True,
        verbose=False,
    )
