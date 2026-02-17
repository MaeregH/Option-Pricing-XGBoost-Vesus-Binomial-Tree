import numpy as np

# def binomial_american(
#     S0: float,    #initial stock price
#     K: float,     #strike price
#     r:  float,    #annual risk-free rate
#     sigma: float, 
#     T: float,     #time to maturity
#     N: int = 100,
#     u = 1.1      #up-factor in binomial model
#     d = 1/u      #recombine trees
#     opttype: str = "C" #differentiate call 'C' or put 'P'

#  ) -> float:
    
#     #deal with expired options

#     if T <= 0:
#         if option_type == "call":
#             return max(S0 - K, 0.0)
#         else:
#             return max(K - S0, 0.0)
        
#         change_t = T/N
#         if change_t <= 0:
#             change_t = 1e-6

    
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
