# Copyright 2019 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# python2 python3
"""Tasks for Spriteworld.

Each class in this file defines a task. Namely, contains a reward function and a
success function for Spriteworld.

The reward function maps an iterable of sprites to a float. The success function
maps an iterable of sprites to a bool.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import numpy as np
import six


@six.add_metaclass(abc.ABCMeta)
class AbstractTask(object):
    """Abstract class from which all tasks should inherit."""

    @abc.abstractmethod
    def reward(self, sprites):
        """Compute reward for the given configuration of sprites.

        This reward is evaluated per-step by the Spriteworld environment. See
        Environment.step() in environment.py for usage. Hence if this is a smooth
        function the agent will have shaped reward. Sparse rewards awarded only at
        the end of an episode can be implemented by returning non-zero reward only
        for a desired goal configuration of sprites (see sub-classes below for
        examples).

        Args:
          sprites: Iterable of sprite instances.

        Returns:
          Float reward for the given configuration of sprites.
        """

    @abc.abstractmethod
    def success(self, sprites):
        """Compute whether the task has been successfully solved.

        Args:
          sprites: Iterable of sprite instances.

        Returns:
          Boolean. Whether or not the given configuration of sprites successfully
            solves the task.
        """


class NoReward(AbstractTask):
    """Used for environments that have no task. Reward is always 0."""

    def __init__(self):
        pass

    def reward(self, unused_sprites):
        """Calculate reward from sprites."""
        return 0.0

    def success(self, unused_sprites):
        return False
