import random
import time
from typing import Callable, Dict
import numpy as np
from pettingzoo.classic import texas_holdem_v4
from pettingzoo.utils import BaseWrapper
from gym.spaces import Discrete


class DeltaEnv(BaseWrapper):
    def __init__(
        self,
        env,
        opponent_choose_move: Callable[[np.ndarray, np.ndarray], int],
        verbose: bool = False,
        render: bool = False,
    ):

        super().__init__(env)
        self.opponent_choose_move = opponent_choose_move
        self.render = render
        self.verbose = verbose

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
        full_space = self.action_space(self.turn)
        if not isinstance(full_space, Discrete):
            raise NotImplementedError(
                "Currently can only parse legal moves for discrete spaces"
            )
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
        if self.turn == self.opponent_agent:
            reward -= self._step(
                self.opponent_choose_move(self.observation, self.legal_moves),
            )

        return self.observation, reward, self.done, self.info

    def print_action(self, action):
        """This doesn't generalise to other card games"""

        player = "Your bot" if self.turn == self.player_agent else "opponent"
        if action == 0:
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

        self.env.step(action)
        reward = self.env.last()[1]

        return reward

    def step(self, move: int):
        # assert (
        #     not self.done
        # ), "Game is done! Call reset() before calling step() again :D"

        reward = self._step(move)

        if not self.done:
            reward -= self._step(
                self.opponent_choose_move(self.observation, self.legal_moves),
            )

        return self.observation, reward, self.done, self.info


def choose_move_randomly(observation, legal_moves):
    if len(legal_moves) == 0:
        return None
    return random.choice(legal_moves)


env = DeltaEnv(texas_holdem_v4.env(), choose_move_randomly, verbose=True, render=True)
observation, reward, done, info = env.reset()
while not done:
    action = choose_move_randomly(observation, info["legal_moves"])

    observation, reward, done, info = env.step(action)
