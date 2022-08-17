import time
from typing import Callable, Dict, Optional

import numpy as np
from gym.spaces import Box, Discrete

from pettingzoo.utils import BaseWrapper


class DeltaEnv(BaseWrapper):
    def __init__(
        self,
        env,
        opponent_choose_move: Callable,
        verbose: bool = False,
        render: bool = False,
        game_speed_multiplier: int = 0,
    ):

        super().__init__(env)
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

        # TODO: Generalise to different games
        self.action_space = Discrete(4)
        self.observation_space = Box(low=0, high=1, shape=(72,))

    @property
    def turn(self) -> str:
        return self.env.agent_selection

    @property
    def observation(self):
        # Not sure all last[0] obs are dicts for all envs?
        return self.env.last()[0]["observation"]

    @property
    def legal_moves(self):
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
        return self.env.last()[2]

    def reset(self):

        super().reset()
        assert len(self.env.agents) == 2, "Two agent game required"
        self.most_recent_move = {self.player_agent: None, self.opponent_agent: None}

        reward = 0
        if self.verbose:
            print("starting game")
        # Take a step if opponent goes first, so step() starts with player
        if self.turn == self.opponent_agent:
            opponent_move = self.opponent_choose_move(self.observation, self.legal_moves)
            reward -= self._step(opponent_move)
        if self.render:
            self.env.render(most_recent_move=self.most_recent_move, render_opponent_cards=False)

        return self.observation, reward, self.done, self.info

    def print_action(self, action):
        """This doesn't generalise to other card games."""

        player = "Player" if self.turn == self.player_agent else "Opponent"
        if action not in self.legal_moves:
            print(f"{player} made an illegal move: {action}!")
        elif action == 0:
            print(f"{player} calls!")
        elif action == 1:
            print(f"{player} raises!")
        elif action == 2:
            print(f"{player} folds!")
        elif action == 3:
            print(f"{player} checks!")

    def _step(self, move: int) -> float:

        assert not self.done, "Game is done! Please reset() the env before calling step() again"
        assert move in self.legal_moves, f"{move} is an illegal move"

        self.most_recent_move[self.env.agent_selection] = move

        if self.verbose:
            self.print_action(move)
            time.sleep(1 / self.game_speed_multiplier)

        self.env.step(move)

        if self.render:
            self.env.render(most_recent_move=self.most_recent_move, render_opponent_cards=False)
            time.sleep(1 / self.game_speed_multiplier)

        reward = self.env.last()[1]

        return reward

    def step(self, move: int):

        reward = self._step(move)

        if not self.done:
            reward = self._step(
                self.opponent_choose_move(self.observation, self.legal_moves),
            )
        if self.done:
            result = "won" if reward > 0 else "lost"
            msg = f"You {result} {abs(reward*2)} chips"

            if self.verbose:
                print(msg)
            if self.render:
                self.env.render(
                    most_recent_move=self.most_recent_move,
                    win_message=msg,
                    render_opponent_cards=True,
                )

        return self.observation, reward, self.done, self.info
