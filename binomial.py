import numpy as np

def binomial_american(
    S0: float,
    K: float,
    r:  float,
    sigma: float,
    T: float,
    N: int = 100,
    option_type: str = "call"

 ) -> float:
    
    #deal with expired options

    if T <= 0:
        if option_type == "call":
            return max(S0 - K, 0.0)
        else:
            return max(K - S0, 0.0)
        
        change_t = T/N
        if change_t <= 0:
            change_t = 1e-6

        

    

    
