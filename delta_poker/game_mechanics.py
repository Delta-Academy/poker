# Hack as this won't pip install on replit
import copy
import random
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import pygame
import torch
from torch import nn

from env_wrapper import BUTTON_DIM, DeltaEnv, get_button_origins
from pettingzoo.classic import texas_holdem_v4

HERE = Path(__file__).parent.resolve()


def choose_move_randomly(state: np.ndarray, legal_moves: np.ndarray):
    return random.choice(legal_moves)


def wait_for_click() -> None:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                return


def play_poker(
    your_choose_move: Callable,
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

    n_hands = 5
    player_chip_change = 0
    for _ in range(n_hands):
        observation, reward, done, info = env.reset()
        while not done:
            action = your_choose_move(observation, info["legal_moves"])
            observation, reward, done, info = env.step(action)
        player_chip_change += reward * 2
        if verbose or render:
            result = "won" if player_chip_change > 0 else "lost"
            print(f"In total you've {result} {abs(player_chip_change)} chips")
        if render:
            wait_for_click()


def PokerEnv(
    opponent_choose_move: Callable[[np.ndarray, np.ndarray], int],
    verbose: bool = False,
    render: bool = False,
    game_speed_multiplier: int = 0,
):
    return DeltaEnv(
        texas_holdem_v4.env(),
        opponent_choose_move,
        verbose,
        render,
        game_speed_multiplier=game_speed_multiplier,
    )


def click_in_button(pos: Tuple[int, int], idx) -> bool:

    x_pos, y_pos = get_button_origins(idx)

    return (
        pos[0] > x_pos
        and pos[0] < x_pos + BUTTON_DIM
        and pos[1] > y_pos
        and pos[1] < y_pos + BUTTON_DIM
    )


LEFT = 1


def human_player(state: np.ndarray, legal_moves: np.ndarray) -> int:
    while True:
        ev = pygame.event.get()
        for event in ev:
            if event.type == pygame.MOUSEBUTTONUP and event.button == LEFT:
                pos = pygame.mouse.get_pos()
                for idx in range(4):
                    if click_in_button(pos, idx) and idx in legal_moves:
                        return idx


class ChooseMoveCheckpoint:
    def __init__(self, checkpoint_name: str, choose_move: Callable):
        self.neural_network = copy.deepcopy(load_checkpoint(checkpoint_name))
        self._choose_move = choose_move

    def choose_move(self, state, legal_moves):
        return self._choose_move(state, legal_moves, self.neural_network)


def checkpoint_model(model: nn.Module, checkpoint_name: str):
    torch.save(model, HERE / checkpoint_name)


def load_checkpoint(checkpoint_name: str):
    return torch.load(HERE / checkpoint_name)


def load_network(team_name: str, network_folder: Path = HERE) -> nn.Module:
    net_path = network_folder / f"{team_name}_network.pt"
    assert (
        net_path.exists()
    ), f"Network saved using TEAM_NAME='{team_name}' doesn't exist! ({net_path})"
    model = torch.load(net_path)
    model.eval()
    return model


def save_network(network: nn.Module, team_name: str) -> None:
    assert isinstance(
        network, nn.Module
    ), f"train() function outputs an network type: {type(network)}"
    assert "/" not in team_name, "Invalid TEAM_NAME. '/' are illegal in TEAM_NAME"
    net_path = HERE / f"{team_name}_network.pt"
    n_retries = 5
    for attempt in range(n_retries):
        try:
            torch.save(network, net_path)
            load_network(team_name)
            return
        except Exception:
            if attempt == n_retries - 1:
                raise
