# Copyright 2026 The cuju Authors
# Copyright 2019 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""GRF's `MultiAgentToSingleAgent` wrapper, over every environment of a batch.

In a scenario that sets `control_all_players`, the policy sees and moves each side's designated
player, the one that leads the play, and the built-in AI moves the others. The guide is in the
repository's `docs/migrating-from-grf.md`, under Scripted players and `control_all_players`.
"""

from __future__ import annotations

from typing import Any

import torch

from .dumps import _CORE_ACTIONS

__all__ = ['MultiAgentToSingleAgent']

_ACTION_COUNTS = {'default': 19, 'v2': 20, 'full': 33}
# GRF: `football_action_set.action_builtin_ai`, index 19 of `v2` and `full`.
_BUILTIN_AI = 19


def _select(value, index):
    """Returns `value[environment, index[environment, k]]` for every environment and `k`."""
    rows = torch.arange(index.shape[0], device=index.device)[:, None]
    if isinstance(value, dict):
        return {name: tensor[rows, index] for name, tensor in value.items()}
    return value[rows, index]


def _broadcast(flags, like):
    """Returns `[environment_count]` flags shaped to broadcast over `like`."""
    return flags.view((-1,) + (1,) * (like.dim() - 1))


class MultiAgentToSingleAgent:
    """GRF's `MultiAgentToSingleAgent` over a batch whose scenario sets `control_all_players`.

    The batch holds an agent for every controllable player of each side the policy plays, and the
    wrapper gives the policy, for each such side, the agent whose active player is the side's
    designated player. Each step moves that agent with the policy's action and every other agent
    with `builtin_ai`, so the batch needs an action set that holds it. `v2` plays GRF's `default`
    set exactly, since `default`'s 19 actions keep their indices in `v2`.

    The reward is the batch's, `[environment_count, agents]`, as GRF's wrapper passes the reward of
    every agent on. GRF's single-agent wrapper then takes the first agent's, as `squeeze` does here.

    Attributes:
        batch: The wrapped batch.
        players: The sides the policy plays, 1 or 2, the left one first.
        action_count: The actions of the policy's action set.
        last_actions: The batch's actions of the last step, `[environment_count, agents]`.
    """

    def __init__(
        self,
        batch: Any,
        left_players: int,
        right_players: int,
        action_set: str | None = None,
        stacked: bool = False,
        squeeze: bool = False,
        _slots: list[int] | None = None,
    ) -> None:
        """Wraps a batch.

        Args:
            batch: A batch with `with_raw=True` or the raw representation, the action set `v2` or
                `full`, which hold `builtin_ai`, no stack, and on each side either no agent or one
                for every controllable player.
            left_players: 1 if the policy plays the left side, else 0.
            right_players: 1 if the policy plays the right side, else 0.
            action_set: The policy's action set, `default`, `v2` or `full`, or `None` for the
                batch's. `default` needs a batch of `v2` or `full`.
            stacked: Whether the observation stacks the last 4 of the policy's observations, as
                GRF's `FrameStack` over this wrapper does. The designated player changes during
                play, so this differs from the batch's own stack, which stacks each agent's.
            squeeze: Whether the observation drops the axis of the sides, except for `raw`, and the
                reward keeps only the first agent's, as GRF's single-agent wrappers do for one side.

        Raises:
            ValueError: If the batch lacks the raw arrays or `builtin_ai`, stacks its observations,
                holds agents on a side that the policy does not play or none on one it plays, or if
                `squeeze` meets two sides or `stacked` meets `raw`.
            KeyError: If `action_set` names no known set.
        """
        if left_players not in (0, 1) or right_players not in (0, 1):
            raise ValueError(
                f'expected 0 or 1 left and right players, found {left_players} and {right_players}'
            )
        if batch.action_count < _ACTION_COUNTS['v2']:
            raise ValueError(
                'expected a batch whose action set holds builtin_ai; create it with '
                "action_set='v2', which plays GRF's default set with the same indices, or 'full'"
            )
        if batch.stacked:
            raise ValueError('expected a batch without a stack; pass stacked=True here instead')
        if getattr(batch, '_raw_fields', None) is None:
            raise ValueError('expected a batch with the raw representation or with_raw=True')
        if stacked and batch._representation == 'raw':
            raise ValueError('expected simple115, simple115v2 or extracted for a stack, found raw')
        # The agents that the policy's sides cover. `create_environment` passes the slots of GRF's
        # `agent` player, and the players of `extra_players` in the other slots act on their own.
        covered = torch.ones(batch.agents, dtype=torch.bool)
        if _slots is not None:
            covered = torch.zeros(batch.agents, dtype=torch.bool)
            covered[list(_slots)] = True
        left = torch.arange(batch.agents) < batch.left_agents[:, None]
        mask = batch.agent_mask & covered
        side_masks = [left & mask, ~left & mask]
        for side, players, agents in (
            ('left', left_players, side_masks[0].sum(1)),
            ('right', right_players, side_masks[1].sum(1)),
        ):
            if bool(((agents > 0) != bool(players)).any()):
                raise ValueError(
                    f'expected {"agents" if players else "no agent"} on the {side} side of every '
                    f'environment, since the policy {"plays" if players else "does not play"} it'
                )
        self.players = left_players + right_players
        if squeeze and self.players != 1:
            raise ValueError('expected one side to squeeze the axis of the sides')
        if action_set is None:
            self.action_count = batch.action_count
        else:
            if action_set not in _ACTION_COUNTS:
                raise KeyError(f'expected the action set {", ".join(map(repr, _ACTION_COUNTS))}')
            self.action_count = _ACTION_COUNTS[action_set]
            if self.action_count > batch.action_count:
                raise ValueError(
                    f'expected a batch with the actions of {action_set!r}, found '
                    f'{batch.action_count}'
                )
        self.batch = batch
        self._sides = [side for side, players in ((0, left_players), (1, right_players)) if players]
        self._stacked = stacked
        self._squeeze = squeeze
        device = batch._torch_device
        self._side_masks = [side_mask.to(device) for side_mask in side_masks]
        self._index = None
        self._frames = None
        self._done = None
        self.last_actions = None

    def _selection(self, raw, only=None):
        """Returns, for each environment, the agent of each side whose active player is designated.

        Args:
            raw: The raw arrays of the batch.
            only: The environments to check, or `None` for every one. The others get any agent.

        Raises:
            RuntimeError: If a checked side has no such agent or several, which GRF's wrapper does
                not handle either.
        """
        match = raw['active'] == raw['designated']
        columns = []
        for side in self._sides:
            found = match & self._side_masks[side]
            wrong = found.sum(1) != 1
            if only is not None:
                wrong &= only
            if bool(wrong.any()):
                environment = int(torch.nonzero(wrong)[0, 0])
                raise RuntimeError(
                    f'environment {environment} has {int(found[environment].sum())} agents on the '
                    f'{("left", "right")[side]} side whose active player is the designated one, '
                    'where the wrapper needs exactly one'
                )
            columns.append(found.int().argmax(1))
        return torch.stack(columns, 1)

    def _shape(self, observation):
        """Applies `squeeze` to an observation of the policy."""
        if not self._squeeze or isinstance(observation, dict):
            # GRF: the raw observation of one agent keeps its agent axis.
            return observation
        return observation.squeeze(1)

    def _current(self):
        """Returns the observation of the selected agents in the last outputs of the batch."""
        return _select(self.batch.observation(), self._index)

    def observation(self) -> Any:
        """Returns the policy's observation of the last `reset` or `step`.

        It is `[environment_count, players, ...]`, or a dictionary for `raw`, stacked and squeezed
        as the options say.
        """
        if self._index is None:
            raise RuntimeError('reset must be called before the observation')
        observation = torch.cat(self._frames, dim=-1) if self._stacked else self._current()
        return self._shape(observation)

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        """Resets the batch, with its arguments, and returns the policy's observation."""
        self.batch.reset(*args, **kwargs)
        self._index = self._selection(self.batch.raw())
        if self._stacked:
            self._frames = [self._current()] * 4
        self._done = None
        return self.observation()

    def _inner_actions(self, actions):
        """Returns the batch's actions, the policy's for the chosen agents and `builtin_ai` else."""
        count = self.action_count
        environments = self.batch.environment_count
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions)
        if actions.is_floating_point() or actions.is_complex():
            raise TypeError(f'expected integer action indices, found {actions.dtype}')
        if self.players == 1 and tuple(actions.shape) == (environments,):
            actions = actions[:, None]
        if tuple(actions.shape) != (environments, self.players):
            raise ValueError(
                f'expected actions of the shape {(environments, self.players)}, found '
                f'{tuple(actions.shape)}'
            )
        # GRF: an action indexes the policy's set, so one outside it raises.
        if actions.device.type == 'cpu' and bool(((actions < -count) | (actions >= count)).any()):
            raise IndexError(f'expected actions from {-count} to {count - 1}')
        actions = actions.to(self._index.device, torch.int64)
        # A negative index counts from the end of the policy's set, not of the batch's.
        actions = torch.where(actions < 0, actions + count, actions)
        inner = torch.full(
            (environments, self.batch.agents), _BUILTIN_AI, dtype=torch.int64, device=actions.device
        )
        inner.scatter_(1, self._index, actions)
        return inner.to(torch.int32)

    def step(
        self, actions: Any, *, _others: list[tuple[int, int]] | None = None
    ) -> tuple[Any, Any, Any, dict[str, Any]]:
        """Runs one step of the batch.

        Args:
            actions: `[environment_count, players]` indices into the policy's action set, or
                `[environment_count]` for one side.

        Returns:
            The policy's observation, the batch's reward `[environment_count, agents]`, or the
            first agent's with `squeeze`, the done flags and the batch's information, whose
            `final_observation` holds the policy's observation of the step that ended each
            episode.

        Raises:
            RuntimeError: Before the first `reset`, or as `_selection`.
        """
        if self._index is None:
            raise RuntimeError('reset must be called before step')
        self.last_actions = self._inner_actions(actions)
        # The players of `extra_players` that `create_environment` adds act in their own slots.
        for slot, action in _others or []:
            self.last_actions[:, slot] = action
        _, reward, done, information = self.batch.step(self.last_actions)
        ended = done.bool().to(self._index.device)
        final = _select(
            information['final_observation'], self._selection(self.batch.final_raw(), ended)
        )
        self._index = self._selection(self.batch.raw())
        if self._stacked:
            current = self._current()
            # The step that ends an episode stacks its final observation, and an automatic reset
            # then fills the stack with the first observation of the next episode.
            frames = [*self._frames[1:], torch.where(_broadcast(ended, current), final, current)]
            final = torch.cat(frames, dim=-1)
            if self.batch.auto_reset:
                frames = [torch.where(_broadcast(ended, current), current, f) for f in frames]
            self._frames = frames
        self._done = ended
        if self._squeeze:
            reward = reward[:, 0]
        return (
            self.observation(),
            reward,
            done,
            {**information, 'final_observation': self._shape(final)},
        )

    def reset_done(self) -> None:
        """Resets the environments whose episode ended, as the batch's `reset_done`."""
        if self._index is None:
            raise RuntimeError('reset must be called before reset_done')
        self.batch.reset_done()
        self._index = self._selection(self.batch.raw())
        if self._stacked and self._done is not None:
            current = self._current()
            ended = _broadcast(self._done, current)
            self._frames = [torch.where(ended, current, frame) for frame in self._frames]
        self._done = None

    def close(self) -> None:
        """Closes the batch."""
        self.batch.close()

    @staticmethod
    def get_observation(observation: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Returns the raw observations of one environment whose active player is designated.

        It is GRF's static method.
        """
        assert 'designated' in observation[0], 'Only raw observations can be converted'
        return [entry for entry in observation if entry['designated'] == entry['active']]

    @staticmethod
    def get_action(actions: Any, original_observation: list[dict[str, Any]]) -> list[Any]:
        """Returns an action for each raw observation of `original_observation`.

        It is GRF's static method. Where the active player is the designated one it takes the next
        of `actions`, and elsewhere GRF's `builtin_ai` action.
        """

        assert 'designated' in original_observation[0], 'Only raw observations can be converted'
        result = [_CORE_ACTIONS['builtin_ai']] * len(original_observation)
        action_index = 0
        for index, entry in enumerate(original_observation):
            if entry['designated'] == entry['active']:
                assert action_index < len(actions)
                result[index] = actions[action_index]
                action_index += 1
        return result
