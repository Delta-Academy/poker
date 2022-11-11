import random
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pygame
import pygame.gfxdraw

from render import draw_chips, render
from rlcard.games.nolimitholdem.game import NolimitholdemGame

BLACK_COLOR = (21, 26, 26)
WHITE_COLOR = (255, 255, 255)
GREY_COLOR = (150, 159, 167)

# Graphics consts
FULL_WIDTH = 400
FULL_HEIGHT = 600

# Table is the game rendered by PettingZoo
TABLE_WIDTH = 375
TABLE_HEIGHT = 500
TABLE_ORIGIN = ((FULL_WIDTH - TABLE_WIDTH) // 2, 0)


MOVE_MAP: Dict[int, str] = {
    0: "fold",
    1: "call\ncheck",
    2: "raise\n",  # Half pot, to be appended with the amount
    3: "raise\n",  # Full pot, to be appended with the amount
    4: "all\nin",
}

# Clickable buttons
N_BUTTONS = len(MOVE_MAP)
GAP_BETWEEN_BUTTONS = 10
BUTTON_MARGIN_HORIZONTAL = (FULL_WIDTH - TABLE_WIDTH) // 2
BUTTON_MARGIN_VERTICAL = 10
BUTTON_DIM = (
    FULL_WIDTH - BUTTON_MARGIN_HORIZONTAL * 2 - N_BUTTONS * GAP_BETWEEN_BUTTONS
) // N_BUTTONS


def get_button_origins(idx: int) -> Tuple[int, int]:
    return (
        BUTTON_MARGIN_HORIZONTAL + idx * (BUTTON_DIM + GAP_BETWEEN_BUTTONS),
        TABLE_ORIGIN[1] + TABLE_HEIGHT + BUTTON_MARGIN_VERTICAL,
    )


class PokerEnv:
    STARTING_MONEY = 100

    def __init__(
        self,
        opponent_choose_move: Callable,
        verbose: bool = False,
        render: bool = False,
        game_speed_multiplier: float = 1.0,
    ):

        self.opponent_choose_move = opponent_choose_move
        self.render = render
        self.verbose = verbose
        self.game_speed_multiplier = game_speed_multiplier

        self.game = NolimitholdemGame()
        self.player_total = self.opponent_total = self.STARTING_MONEY

        # Will be flipped on every hand reset
        self.dealer = random.choice([0, 1])

        # Want to get rid of this
        self.player_agent = 0
        self.opponent_agent = 1

        # self.most_recent_move: Dict[str, Optional[int]] = {
        #     self.player_agent: None,
        #     self.opponent_agent: None,
        # }

        self.env_reset = False
        if render:
            pygame.init()
            self._font = pygame.font.SysFont("arial", 18)
            self._screen = pygame.display.set_mode((FULL_WIDTH, FULL_HEIGHT))
            self.subsurf = self._screen.subsurface(
                (
                    TABLE_ORIGIN[0],
                    TABLE_ORIGIN[1],
                    TABLE_WIDTH,
                    TABLE_HEIGHT,
                )
            )

    @property
    def turn(self) -> int:
        # I THINK
        return self.game.game_pointer

    @property
    def player_state(self) -> Dict:
        # The state is actually quite a nice rich dictioanary here,
        # we could leave it up to users how to represent it?
        return self.game.get_state(self.player_agent)

    @property
    def opponent_state(self) -> Dict:
        return self.game.get_state(self.opponent_agent)

    @property
    def done(self) -> bool:
        return self.game_over

    @property
    def hand_done(self) -> bool:
        # I THINK
        return self.game.is_over()

    # def render_game_tournament(
    #     self, screen: pygame.surface.Surface, win_message: Optional[str]
    # ) -> None:
    #     """Inject a screen and render the table without buttons for the tournament."""

    # self.env.render(
    #     most_recent_move=self.most_recent_move,
    #     render_opponent_cards=True,
    #     win_message=win_message,
    #     screen=screen,
    #     show_player_names=False,
    #     continue_hands=False,
    # )

    def render_game(
        self,
        render_opponent_cards: bool = False,
        win_message: Optional[str] = None,
    ) -> None:

        self._screen.fill((7, 99, 36))  # green background
        render(
            player_states={"player": self.player_state, "opponent": self.opponent_state},
            most_recent_move=self.most_recent_move,
            render_opponent_cards=render_opponent_cards,
            win_message=win_message,
            screen=self.subsurf,
            show_player_names=True,
            continue_hands=not self.game_over,
        )

        # TODO:
        possible_chips = lambda total: min(max(0, total), self.STARTING_MONEY * 2)
        player_chips = int(possible_chips(self.player_total))
        opponent_chips = int(possible_chips(self.opponent_total))

        draw_chips(
            screen=self._screen,
            n_chips=opponent_chips,
            x_pos=0,
            y_pos=int(FULL_HEIGHT * 0.2),
        )
        draw_chips(
            screen=self._screen,
            n_chips=player_chips,
            x_pos=0,
            y_pos=int(FULL_HEIGHT * 0.66),
        )

        self.draw_possible_actions()

        pygame.display.update()
        time.sleep(1 / self.game_speed_multiplier)

    @property
    def legal_moves(self) -> List:
        """Make this is correct for the currently player."""
        return [action.value for action in self.game.get_legal_actions()]

    def draw_possible_actions(self) -> None:

        pot_size = self.player_state["pot"]

        for idx, action in MOVE_MAP.items():
            if idx == 2:
                action += str(pot_size // 2)
            elif idx == 3:
                action += str(pot_size)
            self.draw_action(action, idx, idx in self.legal_moves)

    def draw_action(self, action: str, idx: int, legal: bool) -> None:

        x_pos, y_pos = get_button_origins(idx)

        rect = pygame.Rect(
            x_pos,
            y_pos,
            BUTTON_DIM,
            BUTTON_DIM,
        )
        color = WHITE_COLOR if legal else GREY_COLOR
        pygame.gfxdraw.rectangle(
            self._screen,
            rect,
            color,
        )

        text = self._font.render(action, True, color)

        # Centre the text on the middle of the button
        x_pos += BUTTON_DIM // 2
        text_height = text.get_height() * action.count("\n")
        y_pos += (BUTTON_DIM - text_height) // 2

        self.render_multi_line_centre(action, x_pos, y_pos, self._font.get_height(), color=color)

    @property
    def game_over(self) -> bool:
        return self.player_total <= 0 or self.opponent_total <= 0

    def reset(self) -> Tuple[Dict, float, bool, Dict]:
        """Reset the whole round."""
        self.env_reset = True
        self.player_total = self.opponent_total = self.STARTING_MONEY
        if self.verbose:
            print("New round, resetting chips to starting value")
        return self.reset_hand()

    @property
    def reward(self) -> float:
        # TODO: Henry flagged issue about the STARTING MONEY thing
        if self.player_total <= 0:
            return -self.STARTING_MONEY

        if self.opponent_total <= 0:
            return self.STARTING_MONEY

        try:
            reward = self.game.get_payoffs()[self.player_agent]
        except Exception:
            reward = 0
        return reward

    def reset_hand(self) -> Tuple[Dict, float, bool, Dict]:
        """Reset game to the next hand, persisting chips."""

        # Persist the game over screeen if rendering until reset
        if self.render and self.game_over:
            return self.player_state, 0, True, {}

        self.dealer = int(not self.dealer)

        game_config = {
            "game_num_players": 2,
            "player_chips": [self.player_total, self.opponent_total],
            "dealer_id": self.dealer,
        }

        self.game.configure(game_config)
        self.game.init_game()

        self.most_recent_move: Dict[int, Optional[str]] = {
            self.player_agent: None,
            self.opponent_agent: None,
        }

        # Which elements of the obs vector are in the hand?
        # Probably dont need these variables
        self.hand_idx: Dict[int, List] = {
            self.player_agent: [],
            self.opponent_agent: [],
        }

        if self.verbose:
            print("starting game")

        # Take a step if opponent goes first, so step() starts with player
        if self.turn == self.opponent_agent:
            opponent_move = self.opponent_choose_move(state=self.opponent_state)
            self._step(opponent_move)
            if self.hand_done:
                self.complete_hand(self.reward)

        if self.render:
            # If the opponent folds on the first hand, win message
            win_message = f"You won {int(abs(self.reward))} chips" if self.done else None
            self.render_game(render_opponent_cards=win_message is not None, win_message=win_message)

        return self.player_state, self.reward, self.done, {}

    def print_action(self, action: int) -> None:

        player = "Player" if self.turn == self.player_agent else "Opponent"
        if action not in self.legal_moves:
            print(f"{player} made an illegal move: {action}")
        else:
            print(f"{player} {action}")

    def _step(self, move: int) -> float:

        assert self.env_reset, "You need reset the environment before taking your first step!"
        assert not self.done, "Game is done! Please reset() the env before calling step() again"
        assert move in self.legal_moves, f"{move} is an illegal move"

        self.most_recent_move[self.turn] = move

        if self.verbose:
            self.print_action(move)

        self.game.step(move)

        if self.render:
            self.render_game(render_opponent_cards=False, win_message=None)

    def step(self, move: int) -> Tuple[Dict, float, bool, Dict]:

        self._step(move)

        if not self.hand_done:
            self._step(
                self.opponent_choose_move(state=self.opponent_state),
            )

        if self.hand_done:
            reward = self.complete_hand()
            return self.player_state, reward, self.done, {}

        return self.player_state, self.reward, self.done, {}

    def complete_hand(self) -> float:
        """Finishes a hand and resets, does not reset the whole env as the episod is only over when
        one player runs out of chips.

        Need to store the reward before resetting as this changes self.reward
        """

        # Store as will be changed by self.reset_hand()
        reward = self.reward

        self.player_total += int(reward)
        self.opponent_total -= int(reward)

        if reward == 0:
            win_messsage = "Draw!"
        else:
            result = "won" if reward > 0 else "lost"
            win_messsage = f"You {result} {int(abs(reward))} chips"

        if self.verbose:
            print(win_messsage)

        if self.render:
            self.render_game(render_opponent_cards=True, win_message=win_messsage)
            wait_for_click()

        if not self.game_over:
            self.reset_hand()

        return reward

    def render_multi_line_centre(
        self, string: str, x: int, y: int, fsize: int, color: Tuple[int, int, int]
    ) -> None:
        """Render centre aligned 'string' with line breaks on '\n'."""
        lines = string.splitlines()
        for i, l in enumerate(lines):
            text = self._font.render(l, False, color)
            text_rect = text.get_rect(center=(x, y + fsize * i))
            self._screen.blit(text, text_rect)


def wait_for_click() -> None:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                return
