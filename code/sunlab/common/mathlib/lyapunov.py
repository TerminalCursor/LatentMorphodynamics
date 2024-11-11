def trajectory_to_distances(x):
    """X: [N,N_t,N_d]
    ret [N,N_t]"""
    from numpy import zeros
    from numpy.linalg import norm
    from itertools import product, combinations

    x = [x[idx, ...] for idx in range(x.shape[0])]
    pairwise_trajectories = combinations(x, 2)
    _N_COMB = len(list(pairwise_trajectories))
    N_max = x[0].shape[0]
    distances = zeros((_N_COMB, N_max))
    pairwise_trajectories = combinations(x, 2)
    for idx, (a_t, b_t) in enumerate(pairwise_trajectories):
        distances[idx, :] = norm(a_t[:N_max, :] - b_t[:N_max, :], axis=-1)
    return distances


def Lyapunov_d(X):
    """X: [N,N_t]
    λ_n = ln(|dX_n|/|dX_0|)/n; n = [1,2,...]"""
    from numpy import zeros, log, repeat

    Y = zeros((X.shape[0], X.shape[1] - 1))
    Y = log(X[:, 1:] / repeat([X[:, 0]], Y.shape[1], axis=0).T) / (
        repeat([range(Y.shape[1])], Y.shape[0], axis=0) + 1
    )
    return Y


def Lyapunov_t(X):
    """X: [N,N_t,N_d]"""
    return Lyapunov_d(trajectory_to_distances(X))


Lyapunov = Lyapunov_d


def RelativeDistance_d(X):
    """X: [N,N_t]
    λ_n = ln(|dX_n|/|dX_0|)/n; n = [1,2,...]"""
    from numpy import zeros, log, repeat

    Y = zeros((X.shape[0], X.shape[1] - 1))
    Y = log(X[:, 1:] / repeat([X[:, 0]], Y.shape[1], axis=0).T)
    return Y


def RelativeDistance_t(X):
    """X: [N,N_t,N_d]"""
    return RelativeDistance_d(trajectory_to_distances(X))


RelativeDistance = RelativeDistance_d
