import time
from typing import Callable, Dict

import numpy as np
from gym.spaces import Box, Discrete
from pettingzoo.utils import BaseWrapper, env_logger


class DeltaEnv(BaseWrapper):
    def __init__(
        self,
        env,
        opponent_choose_move: Callable,
        verbose: bool = False,
        render: bool = False,
    ):

        super().__init__(env)
        self.opponent_choose_move = opponent_choose_move
        self.render = render
        self.verbose = verbose

        # TODO: Generalise to different games
        self.action_space = Discrete(4)
        self.observation_space = Box(low=0, high=1, shape=(72,))
        # self.logger = env_logger.EnvLogger.get_logger()
        env_logger.EnvLogger.suppress_output()

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
        # The player to move first is randomised by the env
        self.player_agent = self.env.agents[0]
        self.opponent_agent = self.env.agents[1]
        reward = 0
        if self.verbose:
            print("starting game")
        if self.turn == self.opponent_agent:
            opponent_move = self.opponent_choose_move(self.observation, self.legal_moves)
            # Stable baselines requires that you only return the obs from reset
            # You can get rewarded on the first hand in poker
            # if your opponent folds, but you won't have taken any actions
            # so it's maybe fine not to know
            # For now i've just scrapped this edge case
            if opponent_move == 2:
                if self.verbose:
                    print("edge case resetting")
                return self.reset()

            reward -= self._step(opponent_move)

        return self.observation

    def print_action(self, action):
        """This doesn't generalise to other card games."""

        player = "Your bot" if self.turn == self.player_agent else "Opponent"
        if action not in self.legal_moves:
            print(f"{player} made an illegal mode: {action}!")
        elif action == 0:
            print(f"{player} calls!")
        elif action == 1:
            print(f"{player} raises!")
        elif action == 2:
            print(f"{player} folds!")
        elif action == 3:
            print(f"{player} checks!")

    def _step(self, action: int) -> float:
        if self.render:
            self.env.render()
            time.sleep(1)
        if self.verbose:
            self.print_action(action)

        if not self.done:
            self.env.step(action)
        reward = self.env.last()[1]

        return reward

    def step(self, move: int):

        if move not in self.legal_moves:
            # env only gives -1 for an illegal move,
            # but i think they should be punished more
            reward = -10.0
            self._step(move)
            raise ValueError("illegal move")
        else:
            reward = self._step(move)

        if not self.done:
            reward = self._step(
                self.opponent_choose_move(self.observation, self.legal_moves),
            )
        if self.done and self.verbose:
            print(f"Game over!! Reward {reward}")
            print("\n")

        return self.observation, reward, self.done, self.info
