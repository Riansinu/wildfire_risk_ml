"""Independent numerical check of the Laplace evidence on a simple likelihood."""
import numpy as np
from scipy.integrate import quad
from scipy.special import expit
from scipy.stats import norm
from wildfire.bma import BayesianModelAverage


def test_intercept_evidence_agrees_with_direct_integration():
    # All-zero covariates carry no information: integrating their normalized priors
    # must cancel exactly. Balanced outcomes make a stable one-dimensional reference.
    X = np.zeros((40, 10)); y = np.array([0, 1] * 20)
    model = BayesianModelAverage(prior_scale=1).fit(X, y)
    offset = 40 * np.log(.5)

    def integrand(beta):
        loglik = 20 * np.log(expit(beta)) + 20 * np.log(expit(-beta))
        return np.exp(loglik + norm.logpdf(beta) - offset)

    integral, error = quad(integrand, -15, 15, epsabs=1e-10)
    reference = np.log(integral) + offset
    assert error < 1e-7
    assert abs(model.models_[0]['log_evidence'] - reference) < .02
    np.testing.assert_allclose([m['log_evidence'] for m in model.models_],
                               model.models_[0]['log_evidence'], atol=1e-8)
    np.testing.assert_allclose(model.weights_, 1 / 16, atol=1e-8)
