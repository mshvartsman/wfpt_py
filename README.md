# wfpt_py

Various expressions having to do with first passage times of a Wiener process. Includes expected values of decision time and hitting probability from: 

Bogacz, R., Brown, E., Moehlis, J., Holmes, P., & Cohen, J. D. (2006). The physics of optimal decision making: a formal analysis of models of performance in two-alternative forced-choice tasks. Psychological Review, 113(4), 700–65. http://doi.org/10.1037/0033-295X.113.4.700

Srivastava, V., Feng, S. F., Cohen, J. D., Leonard, N. E., & Shenhav, A. (2015). First Passage Time Properties for Time-varying Diffusion Models: A Martingale Approach. http://arxiv.org/abs/1508.03373

Also includes fast wiener first passage time distributions in python using the series truncations from: 

Navarro, D. J., & Fuss, I. G. (2009). Fast and accurate calculations for first-passage times in Wiener diffusion models. Journal of Mathematical Psychology, 53(4), 222–230. doi:10.1016/j.jmp.2009.02.003

Package is presently experimental and not terribly well-tested. You have been warned! 

# Examples

Set up first. Note that parameters are absolute rather than relative and using the notation of Bogacz/Srivastava rather than Ratcliff. 

```
from wfpt import wfpt
import numpy as np

x0 = -0.6
t0 = 0.3
a = 0.1
z = 1
s = 1
dt = 0.001 # only relevant for drawing random RTs
```

Simulated RTs are very slow and only here for testing. At some point I might speed this up. 

```
simulatedRTs = np.array([wfpt.simulate_wfpt(x0, t0, a, z, dt)[0] for i in range(500)]) 
np.mean(np.abs(simulatedRTs))
wfpt.wfpt_rt(x0, t0, a, z, s)
wfpt.wfpt_er(x0, t0, a, z, s)
wfpt.wfpt_dt_upper(x0, t0, a, z, s)
wfpt.wfpt_dt_lower(x0, t0, a, z, s)
```