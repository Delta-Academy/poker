import time
from typing import Callable, Dict, Optional

import numpy as np
from gym.spaces import Box, Discrete
from pettingzoo.utils import AECEnv, BaseWrapper


class DeltaEnv(BaseWrapper):
    def __init__(
        self,
        env: AECEnv,
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
            assert np.all(np.isin(obs[52:], [0, 1]))

        obs[:52] = cards
        return obs

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

    def render_game(
        self, render_opponent_cards=False, win_message=None, screen=None, player_names=None
    ) -> None:
        self.env.render(
            most_recent_move=self.most_recent_move,
            render_opponent_cards=render_opponent_cards,
            win_message=win_message,
            screen=screen,
            player_names=player_names,
        )
        time.sleep(1 / self.game_speed_multiplier)

    def reset(self):

        super().reset()

        assert len(self.env.agents) == 2, "Two agent game required"
        self.most_recent_move = {self.player_agent: None, self.opponent_agent: None}
        # Which elements of the obs vector are in the hand?
        self.hand_idx = {
            self.player_agent: [],
            self.opponent_agent: [],
        }
        reward = 0
        if self.verbose:
            print("starting game")
        # Take a step if opponent goes first, so step() starts with player
        if self.turn == self.opponent_agent:
            opponent_move = self.opponent_choose_move(
                state=self.observation, legal_moves=self.legal_moves
            )
            reward -= self._step(opponent_move)
        if self.render:
            # If the opponent folds on the first hand, win message
            # win_message = None if not self.done else f"You won {abs(reward*2)} chips"
            win_message = None
            self.render_game(render_opponent_cards=win_message is not None, win_message=win_message)

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

        self.env.step(move)

        if self.render:
            self.render_game(render_opponent_cards=False, win_message=None)

        return self.env.last()[1]

    def step(self, move: int):

        reward = self._step(move)

        if not self.done:
            reward = self._step(
                self.opponent_choose_move(state=self.observation, legal_moves=self.legal_moves),
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
