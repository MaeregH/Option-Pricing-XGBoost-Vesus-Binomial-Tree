import numpy as np
from typing import Literal

def binomial_american(
        S0: float,    #initial stock price
        K: float,     #strike price
        r:  float,    #annual risk-free rate
        sigma: float, 
        T: float,     #time to maturity
        N: int = 100,
        u = 1.1,     #up-factor in binomial model
        d = 1/u,     #recombine trees
        opttype: Literal['C', 'P'] = 'C' #differentiate call 'C' or put 'P'

) -> float:
    
    if T <= 0:
        if opttype == 'C':
            return max(S0-K, 0.0)
        else:
            return max(K-S0, 0.0)
        
    dt = max(T/N, 1e-10)
    u = np.exp(sigma * np.sqrt())
    d = 1/u
    q = (np.exp(r * dt) - d) /(u-d)

    if not(0 <= q <= 1):
        raise ValueError(f"Invalid risk-neutral probability: q={q}")
    
    disc = np.exp(-r * dt)
    S = S0 * d ** np.arrange(N, -1, 1) * u ** np.arrange(0, N, N+1)

    if opttype == 'C':
        C = np.maximum(S-K, 0)
    else:
        C = np.maximum(K-S, 0)
    
    for i in range(N-1, -1, -1):
        S = S0 * d ** np.arrange(i, -1, -1) * u ** np.arrange(0, i+1)
        C[:i+1] = disc * (q*C[1:i + 2] + (1-q) * C[:i + 1])

        if opttype == 'C':
            exercise_value = np.maximum(S-K, 0)
        else:
            exercise_value = np.maximum(K-S, 0)

        C[:i+1] = np.maximum(C[:i+1], exercise_value)

    return C[0]



def binomial_tree_fast(K, T, S0, r, N, u, d, opttype='C'):
    dt = T/N
    q = (np.exp(r*dt) - d) / (u-d)
    disc = np.exp(-r*dt)

    #asset price at time of maturity
    C = S0 * d ** (np.arrange(N, -1, -1)) * u ** (np.arrange(0, N+1, 1))

    for i in np.arrange(N, 0, -1):
        for j in range(0, i):
            C[j] = disc * (q*C[j+1] + (1-q)*C[j])

    return C[0]


def trinomial_american(
        
        S0: float,
        K: float,
        r: float,
        sigma: float,
        T: float,
        N: int = 100,
        opttypetype: Literal['C', 'P'] = 'C'
    ) -> float:

    if T <= 0:
        if opttype == 'C':
            return max(S0-K, 0.0)
        else:
            return max(K-S0, 0.0)
        
    dt = max(T/N, 1e-10)
    u = np.exp(sigma * np.sqrt())
    d = 1/u
    
    pu = ((np.exp(r*dt/2) - np.exp(-sigma * np.sqrt(dt / 2))) / (np.exp(sigma * np.sqrt(dt / 2)) - np.exp(-sigma * np.sqrt(dt / 2)))) ** 2
    pd = ((np.exp(sigma * np.sqrt(dt / 2)) - np.exp(r * dt / 2)) / (np.exp(sigma * np.sqrt(dt / 2)) - np.exp(-sigma * np.sqrt(dt / 2)))) ** 2

    pm = 1 - pu - pd

    if not(0<= pu <= 1 and 0 <= pd <= 1 and 0 <= pm <= 1):
        raise ValueError(f"Invalid probabilities: pu={pu}m pm = {pu}, pd = {pd}")
    
    disc = np.exp(-r* dt)
    option_tree = {}

    for j in range(-N, N+1):
        S = S0 * (u ** max(j, 0)) * (d ** max(-j, 0))

        if opttype == 'C':
            option_tree[(N, j)] = max(S - K, 0)
        else:
            option_tree[(N, j)] = max(K-S, 0)


    for i in range(N -1, -1, -1):
        for j in range(-i, i+1):
            S = S0 * (u ** max(j,0)) * (d ** max(-j, 0))        


