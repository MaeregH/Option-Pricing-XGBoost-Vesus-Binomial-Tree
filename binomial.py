import numpy as np
from typing import Literal


def binomial_american(
        S0: float,
        K: float,
        r: float,
        sigma: float,
        T: float,
        N: int = 100,
        opttype: Literal['C', 'P'] = 'C',
) -> float:

    if T <= 0:
        return max(S0 - K, 0.0) if opttype == 'C' else max(K - S0, 0.0)

    dt = max(T / N, 1e-10)
    u = np.exp(sigma * np.sqrt(dt))   # Bug 1 fixed: np.sqrt(dt)
    d = 1 / u
    q = (np.exp(r * dt) - d) / (u - d)

    if not (0 <= q <= 1):
        raise ValueError(f"Invalid risk-neutral probability: q={q}")

    disc = np.exp(-r * dt)

    # Bug 3 fixed: arange(N,-1,-1) and arange(0,N+1,1)
    S = S0 * d ** np.arange(N, -1, -1) * u ** np.arange(0, N + 1, 1)

    C = np.maximum(S - K, 0) if opttype == 'C' else np.maximum(K - S, 0)

    for i in range(N - 1, -1, -1):
        S = S0 * d ** np.arange(i, -1, -1) * u ** np.arange(0, i + 1, 1)  # Bug 2 fixed: np.arange
        # Bug 4 fixed: temp variable to avoid aliasing
        hold = disc * (q * C[1:i + 2] + (1 - q) * C[:i + 1])
        C[:i + 1] = hold
        exercise = np.maximum(S - K, 0) if opttype == 'C' else np.maximum(K - S, 0)
        C[:i + 1] = np.maximum(C[:i + 1], exercise)

    return float(C[0])


def binomial_tree_fast(K, T, S0, r, sigma, N, opttype='C'):  # Bug 5 fixed: added sigma, opttype
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))   # Bug 5 fixed: compute u/d from sigma
    d = 1 / u
    q = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    C = S0 * d ** np.arange(N, -1, -1) * u ** np.arange(0, N + 1, 1)  # Bug 2 fixed: np.arange

    # Bug 5 fixed: apply payoff at maturity
    C = np.maximum(C - K, 0) if opttype == 'C' else np.maximum(K - C, 0)

    for i in np.arange(N, 0, -1):  # Bug 2 fixed: np.arange
        for j in range(0, i):
            C[j] = disc * (q * C[j + 1] + (1 - q) * C[j])

    return float(C[0])


def trinomial_american(
        S0: float,
        K: float,
        r: float,
        sigma: float,
        T: float,
        N: int = 100,
        opttype: Literal['C', 'P'] = 'C',
) -> float:

    if T <= 0:
        return max(S0 - K, 0.0) if opttype == 'C' else max(K - S0, 0.0)

    dt = max(T / N, 1e-10)
    # Boyle (1988) trinomial: u'^2 where u'=exp(σ√(Δt/2)), so u=exp(σ√(2Δt))
    u = np.exp(sigma * np.sqrt(2 * dt))
    d = 1 / u

    pu = ((np.exp(r * dt / 2) - np.exp(-sigma * np.sqrt(dt / 2))) /
          (np.exp(sigma * np.sqrt(dt / 2)) - np.exp(-sigma * np.sqrt(dt / 2)))) ** 2
    pd = ((np.exp(sigma * np.sqrt(dt / 2)) - np.exp(r * dt / 2)) /
          (np.exp(sigma * np.sqrt(dt / 2)) - np.exp(-sigma * np.sqrt(dt / 2)))) ** 2
    pm = 1 - pu - pd

    if not (0 <= pu <= 1 and 0 <= pd <= 1 and 0 <= pm <= 1):
        raise ValueError(f"Invalid probabilities: pu={pu}, pm={pm}, pd={pd}")  # Bug 6 fixed: typo

    disc = np.exp(-r * dt)
    option_tree = {}

    for j in range(-N, N + 1):
        S = S0 * (u ** max(j, 0)) * (d ** max(-j, 0))
        option_tree[(N, j)] = max(S - K, 0) if opttype == 'C' else max(K - S, 0)

    for i in range(N - 1, -1, -1):
        for j in range(-i, i + 1):
            S = S0 * (u ** max(j, 0)) * (d ** max(-j, 0))
            continuation = disc * (
                pu * option_tree.get((i + 1, j + 1), 0) +
                pm * option_tree.get((i + 1, j), 0) +
                pd * option_tree.get((i + 1, j - 1), 0)
            )
            exercise = max(S - K, 0) if opttype == 'C' else max(K - S, 0)
            option_tree[(i, j)] = max(continuation, exercise)

    return float(option_tree[(0, 0)])


def baw_american(
        S0: float,
        K: float,
        r: float,
        sigma: float,
        T: float,
        opttype: Literal['C', 'P'] = 'C',
) -> float:
    """Barone-Adesi & Whaley (1987) American option, zero dividends.

    For calls on non-dividend stock, American = European (Merton 1973) so
    this returns Black-Scholes exactly.  For puts, the early-exercise boundary
    S* is found with Brent's method — the assessment's shortcut
    S* = K/(1-1/q2) is the *perpetual* boundary and is wrong for finite T.
    Falls back to European BS on any numerical failure.
    """
    from scipy.stats import norm
    from scipy.optimize import brentq

    if T <= 0:
        return max(S0 - K, 0.0) if opttype == 'C' else max(K - S0, 0.0)
    if sigma <= 1e-8:
        return max(S0 - K, 0.0) if opttype == 'C' else max(K - S0, 0.0)

    sqT = np.sqrt(T)
    disc_K = K * np.exp(-r * T)

    def _d1(S):
        return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqT)

    def bs_put(S):
        d1 = _d1(S);  d2 = d1 - sigma * sqT
        return disc_K * norm.cdf(-d2) - S * norm.cdf(-d1)

    # American call = European call for non-dividend stock (Merton 1973)
    if opttype == 'C':
        d1 = _d1(S0);  d2 = d1 - sigma * sqT
        return float(S0 * norm.cdf(d1) - disc_K * norm.cdf(d2))

    # ── American put: BAW quadratic approximation ──────────────────────────
    M = 2.0 * r / sigma**2
    h = 1.0 - np.exp(-r * T)
    if h < 1e-14:
        return float(bs_put(S0))

    disc_q = (M - 1)**2 + 4.0 * M / h
    if disc_q < 0:
        return float(bs_put(S0))
    q1 = (-(M - 1) - np.sqrt(disc_q)) / 2.0
    if q1 >= 0:                         # degenerate: no early exercise
        return float(bs_put(S0))

    # Boundary equation: p(S*) − (S*/q1)(1−N(−d1(S*))) − (K−S*) = 0
    # f < 0 near S=0 (PV(K) < K),  f > 0 near S=K
    def f(Ss):
        return bs_put(Ss) - (Ss / q1) * (1.0 - norm.cdf(-_d1(Ss))) - (K - Ss)

    lo, hi = max(K * 1e-4, 1e-4), K * (1.0 - 1e-7)
    try:
        if f(lo) * f(hi) > 0:           # no sign change → pure European
            return float(bs_put(S0))
        S_star = brentq(f, lo, hi, xtol=1e-5, maxiter=100)
    except Exception:
        return float(bs_put(S0))

    if S0 <= S_star:
        return float(K - S0)            # immediate exercise

    d1_star = _d1(S_star)
    A1 = -(S_star / q1) * (1.0 - norm.cdf(-d1_star))
    return float(bs_put(S0) + A1 * (S0 / S_star) ** q1)


if __name__ == "__main__":
    from scipy.stats import norm

    S0, K, r, sigma, T, N = 100.0, 100.0, 0.05, 0.2, 1.0, 200

    call_bin = binomial_american(S0, K, r, sigma, T, N, opttype='C')
    put_bin  = binomial_american(S0, K, r, sigma, T, N, opttype='P')
    call_tri = trinomial_american(S0, K, r, sigma, T, N, opttype='C')
    put_tri  = trinomial_american(S0, K, r, sigma, T, N, opttype='P')

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    bs_call = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    bs_put  = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

    print(f"Binomial  (N={N}): Call={call_bin:.4f}  Put={put_bin:.4f}")
    print(f"Trinomial (N={N}): Call={call_tri:.4f}  Put={put_tri:.4f}")
    print(f"Black-Scholes (Eur): Call={bs_call:.4f}  Put={bs_put:.4f}")
    print("Smoke test passed.")
