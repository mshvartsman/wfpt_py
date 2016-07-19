import pytest

def test_zeroLik_shortT():
    from wfpt.wfpt import _wfpt_logp
    import numpy as np

    lik = _wfpt_logp(0.2, 1, 0, 0.3, 0.5, 1, eps = 1e-10)

    assert lik == -np.inf
