import unittest
import importlib.util
from pathlib import Path
import sys

import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[1] / "src" / "03_modelo_bayesiano.py"
sys.path.insert(0, str(MODEL_PATH.parent))
SPEC = importlib.util.spec_from_file_location("modelo_bayesiano", MODEL_PATH)
modelo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(modelo)


class TestModeloBayesiano(unittest.TestCase):
    def test_log_likelihood_matches_hand_calculation(self):
        x = np.zeros((3, 6))
        y = np.array([0, 1, 2])
        theta = np.array([0.0, np.log(3), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.assertAlmostEqual(
            modelo.log_likelihood(theta, x, y),
            np.log(1 / 32),
        )

    def test_coefficient_priors_transform_with_train_scales(self):
        scales = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        means, sds, positive = modelo.transformed_coefficient_priors(scales)

        np.testing.assert_allclose(means, [0.06, 0.12, 0.12, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(sds, [0.04, 0.06, 0.08, 1.0, 3.0, 1.0])
        np.testing.assert_array_equal(
            positive, [False, True, True, True, False, False]
        )

    def test_cutpoint_prior_uses_training_class_proportions(self):
        y_train = np.array([0, 0, 1, 2])
        np.testing.assert_allclose(
            modelo.cutpoint_prior_from_training_labels(y_train),
            [0.0, np.log(3)],
        )

    def test_prior_rejects_unordered_cutpoints(self):
        theta = np.array([1.0, 0.0, 0.03, 0.04, 0.03, 0.30, 0.0, 0.0])
        self.assertEqual(
            modelo.log_prior(theta, np.ones(6), [-0.4, 1.0]),
            -np.inf,
        )

    def test_prior_rejects_nonpositive_truncated_coefficient(self):
        theta = np.array([-0.4, 1.0, 0.03, -0.01, 0.03, 0.30, 0.0, 0.0])
        self.assertEqual(
            modelo.log_prior(theta, np.ones(6), [-0.4, 1.0]),
            -np.inf,
        )

    def test_log_posterior_is_likelihood_plus_prior(self):
        scales = np.ones(6)
        cutpoint_means = [-0.4, 1.0]
        x = np.zeros((3, 6))
        y = np.array([0, 1, 2])
        theta = np.array([-0.4, 1.0, 0.03, 0.04, 0.03, 0.30, 0.0, 0.0])

        expected = modelo.log_likelihood(theta, x, y) + modelo.log_prior(
            theta, scales, cutpoint_means
        )
        self.assertAlmostEqual(
            modelo.log_posterior(
                theta, x, y, scales, cutpoint_means
            ),
            expected,
        )

    def test_invalid_shapes_raise_clear_errors(self):
        with self.assertRaises(ValueError):
            modelo.transformed_coefficient_priors([1.0, 2.0])

        with self.assertRaises(ValueError):
            modelo.log_likelihood(np.zeros(8), np.zeros((2, 5)), [0, 1])

        with self.assertRaises(ValueError):
            modelo.cutpoint_prior_from_training_labels([0, 0, 1])


if __name__ == "__main__":
    unittest.main()
