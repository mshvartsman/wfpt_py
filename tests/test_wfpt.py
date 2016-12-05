import pytest

def test_zeroLik_shortT():
    from wfpt.wfpt import wfpt_logp
    import numpy as np

    lik = wfpt_logp(0.2, 1, 0, 0.3, 0.5, 1, eps = 1e-10)

    assert lik == -np.inf


# def test_er_bySim():
#     from wfpt.wfpt import simulate_wfpt, wfpt_er
#     import numpy as np

#     simulatedRTs = np.array([simulate_wfpt(-0.6, 0.3, 0.1, 5)[0] for i in range(5000)])
#     analyticER = wfpt_er(-0.6, 0.3, 0.1, 5, 1)
#     np.sum(simulatedRTs<0)/5000 - analyticRT # similar ballpark? 

# def test_rt_bySim():
#     from wfpt.wfpt import simulate_wfpt, wfpt_rt
#     import numpy as np

#     simulatedRTs = np.array([simulate_wfpt(-0.6, 0.3, 0.1, 5)[0] for i in range(5000)])
#     analyticRT = wfpt_rt(-0.6, 0.3, 0.1, 5, 1)
#     # should be positive and small (overshoot means )
#     assert(0 < (np.mean(np.abs(simulatedRTs)) - analyticRT) < 1) 

