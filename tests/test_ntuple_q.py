"""Tests for NTupleQAgent."""

import numpy as np

from src.agents.ntuple_q import NUM_TUPLES, TUPLE_SHAPES, NTupleQAgent, _board_symmetries


def test_tuple_shapes_are_six_distinct_cells_on_the_board():
    """Each of Matsuzaki's 4 canonical shapes must be 6 distinct in-bounds cells."""
    for shape in TUPLE_SHAPES:
        assert len(shape) == 6
        assert len(set(shape)) == 6
        for row, col in shape:
            assert 0 <= row < 4
            assert 0 <= col < 4


def test_board_symmetries_returns_8_distinct_transforms_of_an_asymmetric_board():
    """A board with no rotational/reflective symmetry of its own must produce
    8 genuinely different transforms - if fewer, the symmetry code has a bug
    (e.g. missing a reflection or double-counting a rotation).
    """
    board = np.arange(16, dtype=np.int32).reshape(4, 4)
    variants = _board_symmetries(board)

    assert len(variants) == 8
    # No two of the 8 transforms should be identical for a fully asymmetric board.
    seen = set()
    for variant in variants:
        seen.add(variant.tobytes())
    assert len(seen) == 8


def test_board_symmetries_of_a_fully_symmetric_board_are_all_equal():
    """A uniform board (every cell the same) is invariant under every
    rotation/reflection - sanity check that symmetries don't corrupt values.
    """
    board = np.full((4, 4), 3, dtype=np.int32)
    variants = _board_symmetries(board)
    for variant in variants:
        assert np.array_equal(variant, board)


def test_q_value_is_symmetric_under_board_rotation():
    """Q(s, a) sums lookups shared across all 8 symmetric readings of a
    tuple, so an untrained agent's Q-value for a board and for any of its
    rotations/reflections must be identical (same LUT entries, just reached
    via a different symmetry index) - this is the entire point of symmetric
    sampling, so it must hold before and after training.
    """
    agent = NTupleQAgent(seed=0)
    board = np.array([[1, 2, 3, 4], [0, 1, 2, 3], [0, 0, 1, 2], [0, 0, 0, 1]], dtype=np.int32)
    rotated = np.rot90(board)

    for action in range(agent.action_size):
        assert agent.q_value(board, action) == agent.q_value(rotated, action)


def test_learn_one_step_td_updates_all_four_tuples():
    """A single 1-step TD update should touch every tuple's table for the
    action taken (all 4 tuples contribute to Q(s, a), so all 4 should gain
    at least one new entry after one learning step on a fresh agent).
    """
    agent = NTupleQAgent(seed=1, n_step=1, learning_rate=0.5, gamma=0.9)
    state = np.array([[1, 2, 3, 4], [0, 1, 2, 3], [0, 0, 1, 2], [0, 0, 0, 1]], dtype=np.int32)
    next_state = state.copy()

    agent.learn(state, action=2, reward=1.0, next_state=next_state, done=False)

    for tuple_idx in range(NUM_TUPLES):
        assert len(agent.tables[2][tuple_idx]) > 0


def test_learn_moves_q_value_toward_target_one_step_td():
    """Starting from an all-zero value function, a single 1-step TD update
    with reward=1, gamma=0, on a terminal transition should move Q(s, a)
    toward the reward itself (target = reward + gamma * 0 * bootstrap = 1),
    the same one-update sanity check used for the other agents.
    """
    agent = NTupleQAgent(seed=2, n_step=1, learning_rate=1.0, gamma=0.0)
    # Fully asymmetric board (every cell distinct) so each tuple's 8
    # symmetric readings hit 8 distinct LUT keys with no accidental
    # collisions - a board with any accidental symmetry would have some
    # symmetries collide onto the same key, receiving the update more than
    # once and making the exact-target assertion below false without that
    # being a bug in the agent.
    state = np.arange(1, 17, dtype=np.int32).reshape(4, 4)

    assert agent.q_value(state, action=0) == 0.0
    agent.learn(state, action=0, reward=1.0, next_state=state, done=True)

    # learning_rate=1.0 with an all-zero start should land exactly on the
    # target (a full step, no partial credit needed since current_q was 0).
    assert np.isclose(agent.q_value(state, action=0), 1.0)


def test_learn_monte_carlo_flushes_full_episode_on_done():
    """n_step=None means full Monte Carlo: on done=True every buffered
    transition gets a no-bootstrap discounted-return target, mirroring
    LinearQAgent/FeatureQAgent's equivalent test.
    """
    agent = NTupleQAgent(seed=3, n_step=None, learning_rate=1.0, gamma=0.9)
    # Each state must be fully asymmetric (see the 1-step TD test above for
    # why) and, since cell values are clamped to 0-15, the three states must
    # be distinct rotations of the same 0-15 permutation so none of their
    # tuple readings collide with each other's LUT keys either (verified:
    # np.roll by 1/2 of arange(16) shares no tuple key with the unrolled
    # version, for all 4 tuple shapes).
    base = np.arange(0, 16, dtype=np.int32)
    states = [
        base.reshape(4, 4),
        np.roll(base, 1).reshape(4, 4),
        np.roll(base, 2).reshape(4, 4),
    ]
    rewards = [1.0, 2.0, 3.0]

    for i in range(2):
        agent.learn(states[i], action=0, reward=rewards[i], next_state=states[i + 1], done=False)
    agent.learn(states[2], action=0, reward=rewards[2], next_state=states[2], done=True)

    # All three transitions should have been flushed with no bootstrap.
    expected_last = rewards[2]
    expected_first = rewards[0] + 0.9 * rewards[1] + 0.9**2 * rewards[2]

    assert np.isclose(agent.q_value(states[2], action=0), expected_last)
    assert np.isclose(agent.q_value(states[0], action=0), expected_first)


def test_q_values_returns_all_actions():
    """q_values(state) must return one value per action, matching the
    q_values(state) -> np.ndarray contract search.py's ValueFunction
    protocol expects (the same shape LinearQAgent.q_values returns).
    """
    agent = NTupleQAgent(seed=4)
    state = np.zeros((4, 4), dtype=np.int32)
    values = agent.q_values(state)
    assert values.shape == (agent.action_size,)


def test_save_and_load_round_trip(tmp_path):
    """Saving and loading should preserve every table entry, n_step, and counters."""
    agent = NTupleQAgent(seed=9, n_step=5)
    state = np.arange(16, dtype=np.int32).reshape(4, 4) % 5
    agent.learn(state, action=3, reward=5.0, next_state=state, done=True)
    agent.epsilon = 0.33
    agent.episodes = 12

    save_path = tmp_path / "ntuple_q.pt"
    agent.save(str(save_path))

    loaded_agent = NTupleQAgent(seed=10)
    loaded_agent.load(str(save_path))

    assert loaded_agent.epsilon == 0.33
    assert loaded_agent.episodes == 12
    assert loaded_agent.steps == 1
    assert loaded_agent.n_step == 5
    assert len(loaded_agent._trajectory) == 0
    for action in range(agent.action_size):
        assert agent.q_value(state, action) == loaded_agent.q_value(state, action)


def test_select_action_respects_valid_actions():
    """Greedy action selection must only ever return an action from the
    provided valid_actions subset.
    """
    agent = NTupleQAgent(seed=5)
    state = np.zeros((4, 4), dtype=np.int32)

    for _ in range(10):
        action = agent.select_action(state, valid_actions=[1, 3], use_epsilon=False)
        assert action in (1, 3)


def test_learn_with_numpy_typed_reward_stores_plain_python_floats():
    """Gym2048Env's "log_score" reward mode returns np.log2(score_gained),
    a numpy.float64, not a plain float - and numpy.float64 IS a subclass of
    the builtin float, so an isinstance(value, float) check silently passes
    for it. This matters because torch.load's weights_only=True unpickler
    (the default since PyTorch 2.6) rejects numpy scalar types outright, so
    a checkpoint trained with this reward mode failed to load with the
    library's own default settings until _update_tables started forcing
    every stored value through float(). Reproduce a numpy-typed reward
    directly (rather than depending on which env reward mode currently
    returns one) and assert every stored value's exact type is the builtin
    float.
    """
    agent = NTupleQAgent(seed=6, n_step=1, learning_rate=0.3, gamma=0.9)
    state = np.arange(1, 17, dtype=np.int32).reshape(4, 4)
    next_state = np.arange(2, 18, dtype=np.int32).reshape(4, 4) % 16

    agent.learn(state, action=1, reward=np.float64(1.0), next_state=next_state, done=False)

    for action_tables in agent.tables:
        for table in action_tables:
            for value in table.values():
                assert type(value) is float, f"expected plain float, got {type(value)}"
