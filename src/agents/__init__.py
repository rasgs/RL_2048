"""RL agents for 2048."""

from .feature_q import FeatureQAgent
from .linear_q import LinearQAgent
from .markov_q import MarkovQAgent
from .replay_buffer import ReplayBuffer

__all__ = ["FeatureQAgent", "LinearQAgent", "MarkovQAgent", "ReplayBuffer"]
