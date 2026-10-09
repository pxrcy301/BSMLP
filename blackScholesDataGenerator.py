import numpy as np
from scipy.special import ndtr

def computeOptionsPrices(sigma: np.ndarray,
                         S: np.ndarray, 
                         K: np.ndarray, 
                         tau: np.ndarray, # tte
                         r: np.ndarray) -> np.ndarray:

    sigma, S, K, tau, r = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (sigma, S, K, tau, r)))

    if np.any(tau < 0):
        raise ValueError("Error in computeOptionsPrices(): tau must be non-negative!")

    discK = K * np.exp(-r*tau)
    volSqrtTau = sigma * np.sqrt(tau)

    # tau == 0 or sigma == 0: price is the discounted intrinsic value (S - K at expiry)
    degenerate = volSqrtTau == 0
    safeVol = np.where(degenerate, 1.0, volSqrtTau)

    d = np.log(S / discK) / safeVol
    d_diff = 0.5 * safeVol
    d_plus = d + d_diff
    d_minus = d - d_diff

    PHI_pos_d_plus = ndtr(d_plus)
    PHI_pos_d_minus = ndtr(d_minus)

    callPrices = S * PHI_pos_d_plus - discK * PHI_pos_d_minus

    return np.where(degenerate, np.maximum(S - discK, 0.0), callPrices)





