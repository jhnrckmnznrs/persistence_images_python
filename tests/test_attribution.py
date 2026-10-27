import numpy as np
import pytest

from persistence_images import Bounds, stable_persistence_image
from persistence_images.attribution import LinearPIModel, explain_linear_pi

BOUNDS = Bounds(-3.0, 2.0, 0.0, 5.0)


def test_reconstructs_original_scaled_linear_prediction():
    rng = np.random.default_rng(15)
    d = np.column_stack([rng.uniform(-3, 2, 19), rng.uniform(.01, 4.8, 19)])
    beta = rng.standard_normal(30)
    center = rng.normal(size=30)
    model = LinearPIModel(beta, 4.1, center=center, scale=2.3)
    r = explain_linear_pi(d, BOUNDS, 6, 5, .5, .7, model,
                          weight_scale=1.8, other_blocks=-.7)
    pi = stable_persistence_image(d, BOUNDS, 6, 5, .5, .7, weight_scale=1.8)
    expected = 4.1 + beta @ ((pi - center) / 2.3) - .7
    np.testing.assert_allclose(r.prediction, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(r.reconstructed_prediction, expected,
                               rtol=1e-12, atol=1e-12)


def test_repeated_points_and_diagonal():
    d = np.array([[0., 1.], [0., 1.], [0., 0.], [-2., 4.]])
    result = explain_linear_pi(d, BOUNDS, 4, 3, 1., 1.,
                               LinearPIModel(np.ones(12), 0.))
    assert result.contributions[0] == result.contributions[1]
    assert result.contributions[2] == 0.


def test_empty_diagram_and_other_blocks():
    r = explain_linear_pi(np.empty((0, 2)), BOUNDS, 4, 3, 1., 1.,
                          LinearPIModel(np.ones(12), 7.), other_blocks=2.)
    assert r.prediction == r.reconstructed_prediction == 9.


def test_vector_scaling_and_invalid_inputs():
    d = np.array([[0., 1.]])
    model = LinearPIModel(np.arange(12.), 1., center=np.zeros(12),
                          scale=np.arange(12.) + 1.)
    explain_linear_pi(d, BOUNDS, 4, 3, 1., 1., model)
    with pytest.raises(ValueError):
        explain_linear_pi(d, BOUNDS, 4, 3, 1., 1.,
                          LinearPIModel(np.ones(11), 0.))
    with pytest.raises(ValueError):
        explain_linear_pi([[1., -.1]], BOUNDS, 4, 3, 1., 1., model)
    with pytest.raises(ValueError):
        explain_linear_pi(d, BOUNDS, 4, 3, 1., 1.,
                          LinearPIModel(np.ones(12), 0., scale=0.))


def test_truncated_gaussian_and_mixed_sign_coefficients():
    d = np.array([[-20., .5], [1., 3.], [1., 3.]])
    coef = np.array([1., -2., 3., -4., 5., -6.])
    a = explain_linear_pi(d, BOUNDS, 2, 3, .5, .5,
                          LinearPIModel(coef, .25))
    assert abs(a.contributions[0]) < 1e-12
    assert a.contributions[1] == a.contributions[2]
