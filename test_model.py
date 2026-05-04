"""Module to perform unit testing on the ML model."""

import unittest
from train import train_model


class TestModel(unittest.TestCase):
    """Test suite for validating machine learning model performance."""

    def test_accuracy(self):
        """
        Verify that the trained model accuracy meets the minimum 
        threshold of 0.5.
        """
        acc = train_model()
        self.assertTrue(acc > 0.5)


if __name__ == "__main__":
    unittest.main()
