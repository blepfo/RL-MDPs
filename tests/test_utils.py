""" Tests for utility functions. """
import unittest

from utils import policy_shape, transitions_shape, history_tuples, history_action_tuples


class TestUtils(unittest.TestCase):
    """ Tests for utility functions. """

    def test_policy_shape(self):
        """ Tests that the policy shape is correct in a variety of settings."""
        self.assertSequenceEqual(
            policy_shape(state_size=2, action_size=2, history_length=1),
            (2, 2))

        self.assertSequenceEqual(
            policy_shape(state_size=2, action_size=3, history_length=1),
            (2, 3))

        self.assertSequenceEqual(
            policy_shape(state_size=3, action_size=2, history_length=1),
            (3, 2))

        self.assertSequenceEqual(
            policy_shape(state_size=2, action_size=2, history_length=4),
            (2, 2, 2, 2, 2))

    def test_transitions_shape(self):
        """ Tests that the transition probabilities shape is correct in a variety of settings."""

        self.assertSequenceEqual(
            transitions_shape(state_size=4, action_size=3, history_length=1),
            (5, 3, 4, 2))

        self.assertSequenceEqual(
            transitions_shape(state_size=4, action_size=3, history_length=2),
            (5, 5, 3, 4, 2))

    def test_history_tuples(self):
        self.assertSequenceEqual(
            list(history_tuples(state_size=2, history_length=1)),
            [(0, ), (1, ), (2, )])

        self.assertSequenceEqual(
            list(history_tuples(state_size=5, history_length=1)),
            [(0, ), (1, ), (2, ), (3, ), (4, ), (5, )])

        self.assertSequenceEqual(
            list(history_tuples(state_size=2, history_length=2)), [(0, 0),
                                                                   (0, 1),
                                                                   (0, 2),
                                                                   (1, 0),
                                                                   (1, 1),
                                                                   (1, 2),
                                                                   (2, 0),
                                                                   (2, 1),
                                                                   (2, 2)])

    def test_history_action_tuples(self):
        self.assertSequenceEqual(
            list(
                history_action_tuples(
                    state_size=2, action_size=3, history_length=1)), [(0, 0),
                                                                      (0, 1),
                                                                      (0, 2),
                                                                      (1, 0),
                                                                      (1, 1),
                                                                      (1, 2),
                                                                      (2, 0),
                                                                      (2, 1),
                                                                      (2, 2)])

        self.assertSequenceEqual(
            list(
                history_action_tuples(
                    state_size=2, action_size=2, history_length=2)),
            [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (0, 2, 0), (0, 2, 1),
             (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1), (1, 2, 0), (1, 2, 1),
             (2, 0, 0), (2, 0, 1), (2, 1, 0), (2, 1, 1), (2, 2, 0), (2, 2, 1)])


if __name__ == '__main__':
    unittest.main()
