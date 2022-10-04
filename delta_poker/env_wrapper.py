import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pygame
import pygame.gfxdraw
from gym.spaces import Box, Discrete

from pettingzoo.utils import AECEnv, BaseWrapper

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

# Clickable buttons
N_BUTTONS = 5
GAP_BETWEEN_BUTTONS = 10
BUTTON_MARGIN_HORIZONTAL = (FULL_WIDTH - TABLE_WIDTH) // 2
BUTTON_MARGIN_VERTICAL = 10
BUTTON_DIM = (
    FULL_WIDTH - BUTTON_MARGIN_HORIZONTAL * 2 - N_BUTTONS * GAP_BETWEEN_BUTTONS
) // N_BUTTONS

MOVE_MAP: Dict[int, str] = {
    0: "fold",
    1: "call\ncheck",
    2: "raise\n",  # Half pot, to be appended with the amount
    3: "raise\n",  # Full pot, to be appended with the amount
    4: "all\nin",
}


def get_button_origins(idx: int) -> Tuple[int, int]:
    return (
        BUTTON_MARGIN_HORIZONTAL + idx * (BUTTON_DIM + GAP_BETWEEN_BUTTONS),
        TABLE_ORIGIN[1] + TABLE_HEIGHT + BUTTON_MARGIN_VERTICAL,
    )


class DeltaEnv(BaseWrapper):
    STARTING_MONEY = 200

    def __init__(
        self,
        env: AECEnv,
        opponent_choose_move: Callable,
        verbose: bool = False,
        render: bool = False,
        game_speed_multiplier: float = 0.0,
    ):

        super().__init__(env)
        self.player_total = self.opponent_total = self.STARTING_MONEY

        self.opponent_choose_move = opponent_choose_move
        self.render = render
        self.verbose = verbose
        self.game_speed_multiplier = game_speed_multiplier

        # The player to move first is randomised by the env
        self.player_agent = "player_0"
        self.opponent_agent = "player_1"

        self.most_recent_move: Dict[str, Optional[int]] = {
            self.player_agent: None,
            self.opponent_agent: None,
        }

        self.action_space = Discrete(5)

        self.observation_space = Box(low=0, high=200, shape=(55,))

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
    def turn(self) -> str:
        return self.env.agent_selection

    @property
    def observation(self) -> np.ndarray:
        # This doesn't generalise across envs
        obs = self.env.last()[0]["observation"]
        cards = obs[:52]

        if len(self.hand_idx[self.turn]) == 0:
            assert np.sum(cards) == 2
            self.hand_idx[self.turn] = list(np.where(cards)[0])
        else:
            cards = -cards
            cards[self.hand_idx[self.turn]] *= -1

        if np.sum(cards != 0) > 2:
            assert sum(cards == 1) == 2
            assert sum(cards == -1) in [3, 4, 5]
            assert list(np.where(cards == 1)[0]) == self.hand_idx[self.turn]

        obs[:52] = cards

        # Add how many chips remaining to the state
        obs = np.append(obs, self.player_total)
        return obs

    @property
    def legal_moves(self) -> np.ndarray:
        mask = self.env.last()[0]["action_mask"]
        full_space = self.action_space
        if not isinstance(full_space, Discrete):
            raise NotImplementedError("Currently can only parse legal moves for discrete spaces")
        return np.arange(full_space.n)[mask.astype(bool)]

    @property
    def info(self) -> Dict:
        return {"legal_moves": self.legal_moves}

    @property
    def done(self) -> bool:
        return self.game_over

    @property
    def hand_done(self) -> bool:
        return self.env.last()[2]

    def render_game(
        self,
        render_opponent_cards: bool = False,
        win_message: Optional[str] = None,
        player_names: Optional[List[str]] = None,
    ) -> None:

        self._screen.fill((7, 99, 36))  # green background
        self.env.render(
            most_recent_move=self.most_recent_move,
            render_opponent_cards=render_opponent_cards,
            win_message=win_message,
            screen=self.subsurf,
            player_names=player_names,
            continue_hands=not self.game_over,
        )

        possible_chips = lambda total: min(max(0, total), self.STARTING_MONEY * 2)

        self.env.env.env.env.draw_chips(
            int(possible_chips(self.opponent_total)), 0, int(FULL_HEIGHT * 0.2)
        )
        self.env.env.env.env.draw_chips(
            int(possible_chips(self.player_total)), 0, int(FULL_HEIGHT * 0.66)
        )
        self.draw_possible_actions()

        pygame.display.update()
        time.sleep(1 / self.game_speed_multiplier)

    def draw_possible_actions(self) -> None:

        pot_size = int(self.observation[52] + self.observation[53])
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

    def reset(self):
        """Reset the whole round."""
        self.player_total = self.opponent_total = self.STARTING_MONEY
        if self.verbose:
            print("New round, resetting chips to starting value")
        return self.reset_hand()

    def reset_hand(self) -> Tuple[np.ndarray, float, bool, Dict]:
        """Reset game to the next hand, persisting chips."""

        if self.render and self.game_over:
            return self.observation, 0, True, self.info

        super().reset(options={"player_chips": [self.player_total, self.opponent_total]})

        assert len(self.env.agents) == 2, "Two agent game required"

        self.most_recent_move = {self.player_agent: None, self.opponent_agent: None}
        # Which elements of the obs vector are in the hand?
        self.hand_idx: Dict[str, List] = {
            self.player_agent: [],
            self.opponent_agent: [],
        }
        reward = 0.0
        if self.verbose:
            print("starting game")

        # Take a step if opponent goes first, so step() starts with player
        if self.turn == self.opponent_agent:
            opponent_move = self.opponent_choose_move(
                state=self.observation, legal_moves=self.legal_moves
            )
            reward = self._step(opponent_move)
            if self.hand_done:
                self.complete_hand(reward)

        if self.render:
            # If the opponent folds on the first hand, win message
            win_message = f"You won {int(abs(reward))} chips" if self.done else None
            self.render_game(render_opponent_cards=win_message is not None, win_message=win_message)

        return self.observation, reward, self.done, self.info
        # return self.observation

    def print_action(self, action: int) -> None:

        player = "Player" if self.turn == self.player_agent else "Opponent"
        if action not in self.legal_moves:
            print(f"{player} made an illegal move: {action}")
        else:
            print(f"{player} {MOVE_MAP[action].strip()}")

    def _step(self, move: int) -> float:

        assert not self.done, "Game is done! Please reset() the env before calling step() again"
        assert move in self.legal_moves, f"{move} is an illegal move"

        self.most_recent_move[self.env.agent_selection] = move

        if self.verbose:
            self.print_action(move)

        self.env.step(move)

        if self.render:
            self.render_game(render_opponent_cards=False, win_message=None)

        return self.env.last()[1]

    def step(self, move: int) -> Tuple[np.ndarray, float, bool, Dict]:

        reward = self._step(move)

        if not self.hand_done:
            reward = self._step(
                self.opponent_choose_move(state=self.observation, legal_moves=self.legal_moves),
            )

        if self.hand_done:
            reward = self.complete_hand(reward)

        return self.observation, reward, self.done, self.info

    def complete_hand(self, reward: float) -> float:
        self.player_total += int(reward)
        self.opponent_total -= int(reward)

        if reward == 0:
            win_messsage = "Draw!"
        else:
            result = "won" if reward > 0 else "lost"
            win_messsage = f"You {result} {int(abs(reward))} chips"

        # If the game is over give a large magnitude reward
        if self.player_total <= 0:
            reward = -self.STARTING_MONEY
        elif self.opponent_total <= 0:
            reward = self.STARTING_MONEY

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
