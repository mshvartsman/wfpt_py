import numpy as np
import math
import logging
from numpy import exp, log, sin, sqrt, pi

logger = logging.getLogger(__name__)

def __stdWfptLargeTime(t, w, nterms):
    # large time expansion from navarro & fuss
    logger.debug("Large time expansion, %i terms" % nterms)
    piSqOv2 = 4.93480220054

    terms = pi * np.fromiter((k * exp(-k**2*t*piSqOv2) * sin (k*pi*w) for k in range(1, nterms)), np.float64)

    return np.sum(terms)

def __stdWfptSmallTime(t, w, nterms):
    # small time expansion navarro & fuss (todo: gondan et al. improvement)
    logger.debug("Small time expansion, %i terms" % nterms)

    fr = -math.floor((nterms-1)/2)
    to = math.ceil((nterms-1)/2)

    terms = 1 / sqrt(2*pi*t**3) * np.fromiter(((w+2*k)*exp(-((w+2*k)**2)/(2*t)) for k in range(fr, to)), np.float64)
    return np.sum(terms)

def wfpt_logp(t, c, x0, t0, a, z, eps = 1e-10):

    # boundary sep is 2 * thresh
    boundarySep = 2 * z
    # initial condition is absolute, so to make it relative divide by boundarySep
    x0tilde = ((x0+z) / boundarySep)
    # normalize time
    ttilde = (t - t0) / (boundarySep**2)

    #if t is below NDT, or x0 is outside bounds, probability of any rt is 0
    if ttilde < 0 or x0 > z or x0 < -z: 
        return -np.inf

    if c == 1: # by default return hit of lower bound, so if resp is correct flip
        a = -a
        x0tilde = 1 - x0tilde


    largeK = np.int(sqrt((-2*log(pi*ttilde*eps))/(pi**2*ttilde)))
    smallK = 2 + sqrt(-2*ttilde*log(2*eps*sqrt(2*pi*ttilde))) 
    if largeK < smallK :
        # make sure eps is small enough for for bound to be valid
        if eps > (1 / (2*sqrt(2*pi*ttilde))):
            smallK = 2 
        p = __stdWfptLargeTime(ttilde, x0tilde, math.ceil(largeK))
    else:
        # make sure eps is small enough for for bound to be valid
        if eps > (1 / (pi*sqrt(ttilde))):
            smallK = math.ceil((1 / (pi*sqrt(t))))
        p = __stdWfptSmallTime(ttilde, x0tilde, math.ceil(smallK))
        
    # scale from the std case to whatever is our actual
    scaler = (1 / boundarySep**2) * exp(-a*boundarySep*x0tilde-(a**2*t/2)) 

    return log(scaler*p)

def __simulate_wfpt_single(x0, t0, a, z):
    particle = x0
    t = 0; 
    while abs(particle) < z:
        particle = particle + np.random.normal(a)
        t = t + 1
    return t0 + t if particle > z else -t0 - t

def simulate_wfpt(x0, t0, a, z):
    return np.fromiter((__simulate_wfpt_single(_x0, _t0, _a, _z) for _x0, _t0, _a, _z in zip(x0, t0, a, z)), np.float64)