from matplotlib import pyplot as plt


def blank_plot(_plt=None, _xticks=False, _yticks=False):
    """# Remove Plot Labels"""
    if _plt is None:
        _plt = plt
    _plt.xlabel("")
    _plt.ylabel("")
    _plt.title("")
    tick_params = {
        "which": "both",
        "bottom": _xticks,
        "left": _yticks,
        "right": False,
        "labelleft": False,
        "labelbottom": False,
    }
    _plt.tick_params(**tick_params)
    for child in plt.gcf().get_children():
        if child._label == "<colorbar>":
            child.set_yticks([])
    axs = _plt.gcf().get_axes()
    try:
        axs = axs.flatten()
    except:
        ...
    for ax in axs:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        ax.tick_params(**tick_params)


def save_plot(name, _plt=None, _xticks=False, _yticks=False, tighten=True):
    """# Save Plot in Multiple Formats"""
    assert type(name) == str, "Name must be string"
    from os.path import dirname
    from os import makedirs

    makedirs(dirname(name), exist_ok=True)
    if _plt is None:
        from matplotlib import pyplot as plt
        _plt = plt
    _plt.savefig(name + ".png", dpi=1000)
    blank_plot(_plt, _xticks=_xticks, _yticks=_yticks)
    if tighten:
        from matplotlib import pyplot as plt
        plt.tight_layout()
    _plt.savefig(name + ".pdf")
    _plt.savefig(name + ".svg")


def scatter_2d(data_2d, labels=None, _plt=None, **matplotlib_kwargs):
    """# Scatter 2d Data

    - data_2d: 2d-dataset to plot
    - labels: labels for each case
    - _plt: Optional specific plot to transform"""
    from .colors import Pcolor

    if _plt is None:
        _plt = plt

    def _filter(data, labels=None, _filter_on=None):
        if labels is None:
            return data, False
        else:
            return data[labels == _filter_on], True

    for _class in [2, 3, 1, 0]:
        local_data, has_color = _filter(data_2d, labels, _class)
        if has_color:
            _plt.scatter(
                local_data[:, 0],
                local_data[:, 1],
                color=Pcolor[_class],
                **matplotlib_kwargs
            )
        else:
            _plt.scatter(local_data[:, 0], local_data[:, 1], **matplotlib_kwargs)
            break
    return _plt


def plot_contour(two_d_mask, color="black", color_map=None, raise_error=False):
    """# Plot Contour of Mask"""
    from matplotlib.pyplot import contour
    from numpy import mgrid

    z = two_d_mask
    x, y = mgrid[: z.shape[1], : z.shape[0]]
    if color_map is not None:
        try:
            color = color_map(color)
        except Exception as e:
            if raise_error:
                raise e
    try:
        contour(x, y, z.T, colors=color, linewidth=0.5)
    except Exception as e:
        if raise_error:
            raise e


def gaussian_smooth_plot(
    xy,
    sigma=0.1,
    extent=[-1, 1, -1, 1],
    grid_n=100,
    grid=None,
    do_normalize=True,
):
    """# Plot Data with Gaussian Smoothening around point"""
    from numpy import array, ndarray, linspace, meshgrid, zeros_like
    from numpy import pi, sqrt, exp
    from numpy.linalg import norm

    if not type(xy) == ndarray:
        xy = array(xy)
    if grid is not None:
        XY = grid
    else:
        X = linspace(extent[0], extent[1], grid_n)
        Y = linspace(extent[2], extent[3], grid_n)
        XY = array(meshgrid(X, Y)).T
    smoothed = zeros_like(XY[:, :, 0])
    factor = 1
    if do_normalize:
        factor = (sqrt(2 * pi) * sigma) ** 2.
    if len(xy.shape) == 1:
        smoothed = exp(-((norm(xy - XY, axis=-1) / (sqrt(2) * sigma)) ** 2.0)) / factor
    else:
        try:
            from tqdm.notebook import tqdm
        except Exception:

            def tqdm(x):
                return x

        for i in tqdm(range(xy.shape[0])):
            if xy.shape[-1] == 2:
                smoothed += (
                    exp(-((norm((xy[i, :] - XY), axis=-1) / (sqrt(2) * sigma)) ** 2.0))
                    / factor
                )
            elif xy.shape[-1] == 3:
                smoothed += (
                    exp(-((norm((xy[i, :2] - XY), axis=-1) / (sqrt(2) * sigma)) ** 2.0))
                    / factor
                    * xy[i, 2]
                )
    return smoothed, XY


def plot_grid_data(xy_grid, data_grid, cbar=False, _plt=None, _cmap="RdBu", grid_min=1):
    """# Plot Gridded Data"""
    from numpy import nanmin, nanmax
    from matplotlib.colors import TwoSlopeNorm

    if _plt is None:
        _plt = plt
    norm = TwoSlopeNorm(
        vmin=nanmin([-grid_min, nanmin(data_grid)]),
        vcenter=0,
        vmax=nanmax([grid_min, nanmax(data_grid)]),
    )
    _plt.pcolor(xy_grid[:, :, 0], xy_grid[:, :, 1], data_grid, cmap="RdBu", norm=norm)
    if cbar:
        _plt.colorbar()


def plot_winding(xy_grid, winding_grid, cbar=False, _plt=None):
    plot_grid_data(xy_grid, winding_grid, cbar=cbar, _plt=_plt, grid_min=1e-5)


def plot_vorticity(xy_grid, vorticity_grid, cbar=False, save=False, _plt=None):
    plot_grid_data(xy_grid, vorticity_grid, cbar=cbar, _plt=_plt, grid_min=1e-1)


plt.blank = lambda: blank_plot(plt)
plt.scatter2d = lambda data, labels=None, **matplotlib_kwargs: scatter_2d(
    data, labels, plt, **matplotlib_kwargs
)
plt.save = save_plot


def interpolate_points(df, columns=["Latent-0", "Latent-1"], kind="quadratic", N=500):
    """# Interpolate points"""
    from scipy.interpolate import interp1d
    import numpy as np

    points = df[columns].to_numpy()
    distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
    distance = np.insert(distance, 0, 0) / distance[-1]
    interpolator = interp1d(distance, points, kind=kind, axis=0)
    alpha = np.linspace(0, 1, N)
    interpolated_points = interpolator(alpha)
    return interpolated_points.reshape(-1, 1, 2)


def plot_trajectory(
    df,
    Fm=24,
    FM=96,
    interpolate=False,
    interpolation_kind="quadratic",
    interpolation_N=500,
    columns=["Latent-0", "Latent-1"],
    frame_column="Frames",
    alpha=0.8,
    lw=4,
    colormap='plasma',
    _plt=None,
    _z=None,
):
    """# Plot Trajectories

    Interpolation possible"""
    import numpy as np
    from matplotlib.collections import LineCollection
    import matplotlib as mpl

    if _plt is None:
        _plt = plt
    if type(_plt) == mpl.axes._axes.Axes:
        _ca = _plt
    else:
        try:
            _ca = _plt.gca()
        except:
            _ca = _plt
    X = df[columns[0]]
    Y = df[columns[1]]
    fm, fM = np.min(df[frame_column]), np.max(df[frame_column])

    if interpolate:
        if interpolation_kind == "cubic":
            if len(df) <= 3:
                return
        elif interpolation_kind == "quadratic":
            if len(df) <= 2:
                return
        else:
            if len(df) <= 1:
                return
        points = interpolate_points(
            df, kind=interpolation_kind, columns=columns, N=interpolation_N
        )
    else:
        points = np.array([X, Y]).T.reshape(-1, 1, 2)

    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(
        segments,
        cmap=plt.get_cmap(colormap),
        norm=mpl.colors.Normalize(Fm, FM),
    )
    if _z is not None:
        from mpl_toolkits.mplot3d.art3d import line_collection_2d_to_3d

        if interpolate:
            _z = _z  # TODO: Interpolate
        line_collection_2d_to_3d(lc, _z)
    lc.set_array(np.linspace(fm, fM, points.shape[0]))
    lc.set_linewidth(lw)
    lc.set_alpha(alpha)
    _ca.add_collection(lc)
    _ca.autoscale()
    _ca.margins(0.04)
    return lc
