def get_levy_flight(T=50, D=2, t0=0.1, alpha=3, periodic=False):
    from numpy import vstack
    from mistree import get_levy_flight as get_flight

    if D == 2:
        x, y = get_flight(T, mode="2D", periodic=periodic, t_0=t0, alpha=alpha)
        xy = vstack([x, y]).T
    elif D == 3:
        x, y, z = get_flight(T, mode="3D", periodic=periodic, t_0=t0, alpha=alpha)
        xy = vstack([x, y, z]).T
    else:
        raise ValueError(f"Dimension {D} not supported!")
    return xy


def get_levy_flights(N=10, T=50, D=2, t0=0.1, alpha=3, periodic=False):
    from numpy import moveaxis, array

    trajectories = []
    for _ in range(N):
        xy = get_levy_flight(T=T, D=D, t0=t0, alpha=alpha, periodic=periodic)
        trajectories.append(xy)
    return moveaxis(array(trajectories), 0, 1)


def get_jitter_levy_flights(
    N=10, T=50, D=2, t0=0.1, alpha=3, periodic=False, noise=5e-2
):
    from numpy.random import randn

    trajectories = get_levy_flights(
        N=N, T=T, D=D, t0=t0, alpha=alpha, periodic=periodic
    )
    return trajectories + randn(*trajectories.shape) * noise


def get_gaussian_random_walk(T=50, D=2, R=5, step_size=0.5, soft=None):
    from numpy import array, sin, cos, exp, zeros, pi
    from numpy.random import randn, uniform, rand
    from numpy.linalg import norm

    def is_in(x, R=1):
        from numpy.linalg import norm

        return norm(x) < R

    X = zeros((T, D))
    for t in range(1, T):
        while True:
            if D == 2:
                angle = uniform(0, pi * 2)
                step = randn(1) * step_size
                X[t, :] = X[t - 1, :] + array([cos(angle), sin(angle)]) * step
            else:
                X[t, :] = X[t - 1, :] + randn(D) / D * step_size
            if soft is None:
                if is_in(X[t, :], R):
                    break
            elif rand() < exp(-(norm(X[t, :]) - R) * soft):
                break
    return X


def get_gaussian_random_walks(N=10, T=50, D=2, R=5, step_size=0.5, soft=None):
    from numpy import moveaxis, array

    trajectories = []
    for _ in range(N):
        xy = get_gaussian_random_walk(T=T, D=D, R=R, step_size=step_size, soft=soft)
        trajectories.append(xy)
    return moveaxis(array(trajectories), 0, 1)


def get_gaussian_sample(T=50, D=2):
    from numpy.random import randn

    return randn(T, D)


def get_gaussian_samples(N=10, T=50, D=2, R=5, step_size=0.5):
    from numpy.random import randn

    return randn(T, N, D)
