 pu = ((np.exp(r * dt / 2) - np.exp(-sigma * np.sqrt(dt / 2))) / 
          (np.exp(sigma * np.sqrt(dt / 2)) - np.exp(-sigma * np.sqrt(dt / 2)))) ** 2
    pd = ((np.exp(sigma * np.sqrt(dt / 2)) - np.exp(r * dt / 2)) / 
          (np.exp(sigma * np.sqrt(dt / 2)) - np.exp(-sigma * np.sqrt(dt / 2)))) ** 2
    pm = 1 - pu - pd
    
    if not (0 <= pu <= 1 and 0 <= pd <= 1 and 0 <= pm <= 1):
        raise ValueError(f"Invalid probabilities: pu={pu}, pm={pm}, pd={pd}")
    
    disc = np.exp(-r * dt)
    option_tree = {}
    
    for j in range(-N, N + 1):
        S = S0 * (u ** max(j, 0)) * (d ** max(-j, 0))
        if option_type == 'C':
            option_tree[(N, j)] = max(S - K, 0)
        else:
            option_tree[(N, j)] = max(K - S, 0)
    
    for i in range(N - 1, -1, -1):
        for j in range(-i, i + 1):
            S = S0 * (u ** max(j, 0)) * (d ** max(-j, 0))
            continuation = disc * (
                pu * option_tree.get((i + 1, j + 1), 0) +
                pm * option_tree.get((i + 1, j), 0) +
                pd * option_tree.get((i + 1, j - 1), 0)
            )
            if option_type == 'C':
                exercise = max(S - K, 0)
            else:
                exercise = max(K - S, 0)
            option_tree[(i, j)] = max(continuation, exercise)
    
    return option_tree[(0, 0)]