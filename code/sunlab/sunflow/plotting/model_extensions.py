from matplotlib import pyplot as plt
from sunlab.common.data.shape_dataset import ShapeDataset
from sunlab.globals import DIR_ROOT


def get_nonphysical_masks(
    model,
    xrange=[-1, 1],
    yrange=[-1, 1],
    bins=[500, 500],
    equivdiameter_threshold=10,
    solidity_threshold=0.1,
    area_threshold=100,
    perimeter_threshold=10,
    area_max_threshold=7000,
    perimeter_max_threshold=350,
    area_min_threshold=100,
    perimeter_min_threshold=5,
    consistency_check=False,
):
    """# Generate the Nonphysical Masks in Grid for Model

    Hard Constraints:
    - Non-negative values
    - Ratios no greater than 1

    Soft Constraints:
    - Area/ Perimeter Thresholds"""
    import numpy as np

    x = np.linspace(xrange[0], xrange[1], bins[0])
    y = np.linspace(yrange[0], yrange[1], bins[1])
    X, Y = np.meshgrid(x, y)
    X, Y = X.reshape((bins[0], bins[1], 1)), Y.reshape((bins[0], bins[1], 1))
    XY = np.concatenate([X.reshape((-1, 1)), Y.reshape((-1, 1))], axis=-1)
    dec_v = model.decoder(XY).numpy().reshape((bins[0] * bins[1], 13))
    lXY = model.scaler.scaler.inverse_transform(dec_v).reshape((bins[0], bins[1], 13))
    # Hard Limits
    non_negative_mask = np.all(lXY > 0, axis=-1)
    solidity_mask = np.abs(lXY[:, :, 6]) <= 1
    extent_upper_bound_mask = lXY[:, :, 7] <= 1
    # Soft Extremas
    area_max_mask = lXY[:, :, 4] < area_max_threshold
    perimeter_max_mask = lXY[:, :, 9] < perimeter_max_threshold
    area_min_mask = lXY[:, :, 4] > area_min_threshold
    perimeter_min_mask = lXY[:, :, 9] > perimeter_min_threshold
    # Self-Consistency
    man_solidity_mask = np.abs(lXY[:, :, 0] / lXY[:, :, 4]) <= 1
    equivalent_diameter_mask = (
        np.abs(lXY[:, :, 5] - np.sqrt(4 * np.abs(lXY[:, :, 0]) / np.pi))
        < equivdiameter_threshold
    )
    convex_area_mask = lXY[:, :, 0] < lXY[:, :, 4] + area_threshold
    convex_perimeter_mask = lXY[:, :, 9] < lXY[:, :, 8] + perimeter_threshold
    mask_info = {
        "non-negative": non_negative_mask,
        "solidity": solidity_mask,
        "extent-max": extent_upper_bound_mask,
        #
        "area-max": area_max_mask,
        "perimeter-max": perimeter_max_mask,
        "area-min": area_min_mask,
        "perimeter-min": perimeter_min_mask,
        #
        "computed-solidity": man_solidity_mask,
        "equivalent-diameter": equivalent_diameter_mask,
        "convex-area": convex_area_mask,
        "convex-perimeter": convex_perimeter_mask,
    }
    if not consistency_check:
        mask_info = {
            "non-negative": non_negative_mask,
            "solidity": solidity_mask,
            "extent-max": extent_upper_bound_mask,
            #
            "area-max": area_max_mask,
            "perimeter-max": perimeter_max_mask,
            "area-min": area_min_mask,
            "perimeter-min": perimeter_min_mask,
        }
    mask_list = [mask_info[key] for key in mask_info.keys()]
    return np.all(mask_list, axis=0), X, Y, mask_info


def excavate(input_2d_array):
    """# Return Boundaries for Masked Array

    Use X, Y directions only"""
    from copy import deepcopy as dc
    from numpy import nan_to_num, zeros_like, abs

    data_2d_array = dc(input_2d_array)
    data_2d_array = nan_to_num(data_2d_array, nan=20)
    # X-Gradient
    x_grad = zeros_like(data_2d_array)
    x_grad[:-1, :] = data_2d_array[1:, :] - data_2d_array[:-1, :]
    x_grad[(abs(x_grad) > 10)] = 10
    x_grad[(abs(x_grad) < 10) & (abs(x_grad) > 0)] = 1
    x_grad[x_grad == 1] = 0.5
    x_grad[x_grad > 1] = 1
    # Y-Gradient
    y_grad = zeros_like(data_2d_array)
    y_grad[:, :-1] = data_2d_array[:, 1:] - data_2d_array[:, :-1]
    y_grad[(abs(y_grad) > 10)] = 10
    y_grad[(abs(y_grad) < 10) & (abs(y_grad) > 0)] = 1
    y_grad[y_grad == 1] = 0.5
    y_grad[y_grad > 1] = 1
    return x_grad, y_grad


def excavate_extra(input_2d_array, N=1):
    """# Return Boundaries for Masked Array

    Use all 8 directions"""
    from copy import deepcopy as dc
    from numpy import nan_to_num, zeros_like, abs

    data_2d_array = dc(input_2d_array)
    data_2d_array = nan_to_num(data_2d_array, nan=20)
    # X-Gradient
    x_grad = zeros_like(data_2d_array)
    x_grad[:-N, :] = data_2d_array[N:, :] - data_2d_array[:-N, :]
    x_grad[(abs(x_grad) > 10)] = 10
    x_grad[(abs(x_grad) < 10) & (abs(x_grad) > 0)] = 1
    x_grad[x_grad == 1] = 0.5
    x_grad[x_grad > 1] = 1
    # Y-Gradient
    y_grad = zeros_like(data_2d_array)
    y_grad[:, :-N] = data_2d_array[:, N:] - data_2d_array[:, :-N]
    y_grad[(abs(y_grad) > 10)] = 10
    y_grad[(abs(y_grad) < 10) & (abs(y_grad) > 0)] = 1
    y_grad[y_grad == 1] = 0.5
    y_grad[y_grad > 1] = 1
    # XY-Gradient
    xy_grad = zeros_like(data_2d_array)
    xy_grad[:-N, :-N] = data_2d_array[N:, N:] - data_2d_array[:-N, :-N]
    xy_grad[(abs(xy_grad) > 10)] = 10
    xy_grad[(abs(xy_grad) < 10) & (abs(xy_grad) > 0)] = 1
    xy_grad[xy_grad == 1] = 0.5
    xy_grad[xy_grad > 1] = 1
    # X(-Y)-Gradient
    yx_grad = zeros_like(data_2d_array)
    yx_grad[:-N, :-N] = data_2d_array[N:, :-N] - data_2d_array[:-N, N:]
    yx_grad[(abs(yx_grad) > 10)] = 10
    yx_grad[(abs(yx_grad) < 10) & (abs(yx_grad) > 0)] = 1
    yx_grad[yx_grad == 1] = 0.5
    yx_grad[yx_grad > 1] = 1
    xyn_grad = dc(yx_grad)
    # (-X)Y-Gradient
    xny_grad = zeros_like(data_2d_array)
    xny_grad[:-N, :-N] = data_2d_array[:-N, N:] - data_2d_array[N:, :-N]
    xny_grad[(abs(xy_grad) > 10)] = 10
    xny_grad[(abs(xy_grad) < 10) & (abs(xy_grad) > 0)] = 1
    xny_grad[xy_grad == 1] = 0.5
    xny_grad[xy_grad > 1] = 1
    # (-X)(-Y)-Gradient
    xnyn_grad = zeros_like(data_2d_array)
    xnyn_grad[:-N, :-N] = data_2d_array[:-N, :-N] - data_2d_array[N:, N:]
    xnyn_grad[(abs(yx_grad) > 10)] = 10
    xnyn_grad[(abs(yx_grad) < 10) & (abs(yx_grad) > 0)] = 1
    xnyn_grad[yx_grad == 1] = 0.5
    xnyn_grad[yx_grad > 1] = 1
    return x_grad, y_grad, xy_grad, xyn_grad, xny_grad, xnyn_grad


def excavate_outline(arr, thickness=1):
    """# Generate Transparency Mask with NaNs"""
    from numpy import sum, abs, NaN

    outline = sum(abs(excavate_extra(arr, thickness)), axis=0)
    outline[outline == 0] = NaN
    outline[outline > 0] = 0
    return outline


def get_boundary_outline(
    aae_model_object,
    pixel_classification_file=None,
    include_transition_regions=False,
    border_thickness=3,
    bin_count=800,
    xrange=[-6.5, 6.5],
    yrange=[-4.5, 4.5],
    threshold=0.75,
):
    """# Get Boundary Outlines"""
    from copy import deepcopy
    import numpy as np

    if pixel_classification_file is None:
        pixel_classification_file = "../../extra_data/PhenotypePixels_65x45_800.npy"
    base_classification = np.loadtxt(pixel_classification_file)
    base_classification = base_classification.reshape((bin_count, bin_count, 4))
    max_classification_probability = np.zeros((bin_count, bin_count, 1))
    max_classification_probability[:, :, 0] = (
        np.max(base_classification, axis=-1) < threshold
    )
    classes_with_include_transition_regions = np.concatenate(
        [base_classification, max_classification_probability], axis=-1
    )
    if include_transition_regions:
        phenotype_probabilities = deepcopy(
            np.argsort(classes_with_include_transition_regions[:, :, :], axis=-1)[
                :, :, -1
            ]
        ).astype(np.float32)
    else:
        phenotype_probabilities = deepcopy(
            np.argsort(classes_with_include_transition_regions[:, :, :-1], axis=-1)[
                :, :, -1
            ]
        ).astype(np.float32)
    nonphysical_mask, _, _, _ = get_nonphysical_masks(
        aae_model_object, xrange=xrange, yrange=yrange, bins=[bin_count, bin_count]
    )
    nonphysical_mask = nonphysical_mask.astype(np.float32)
    nonphysical_mask[nonphysical_mask == 0] = np.NaN
    nonphysical_mask[nonphysical_mask == 1] = 0
    nonphysical_mask = nonphysical_mask.T
    phenotype_regions = deepcopy(phenotype_probabilities.T + nonphysical_mask.T)
    outline = excavate_outline(phenotype_regions, border_thickness)
    return outline


def apply_boundary(
    model_loc=DIR_ROOT + "models/current_model/",
    border_thickness=3,
    include_transition_regions=False,
    threshold=0.7,
    alpha=1,
    _plt=None,
):
    """# Apply Boundary to Plot

    Use Pregenerated Boundary by Default for Speed"""
    from ..models import load_aae
    from sunlab.common.scaler import MaxAbsScaler
    import numpy as np

    if _plt is None:
        _plt = plt
    if (model_loc == model_loc) and (border_thickness == 3) and (threshold == 0.7):
        XYM = np.load(DIR_ROOT + "extra_data/OutlineXYM.npy")
        XY = XYM[:2, :, :]
        if include_transition_regions:
            outline = XYM[3, :, :]
        else:
            outline = XYM[2, :, :]
        _plt.pcolor(XY[0, :, :], XY[1, :, :], outline, cmap="gray", alpha=alpha)
        return
    model = load_aae(model_loc, MaxAbsScaler)
    bin_count = 800
    xrange = [-6.5, 6.5]
    yrange = [-4.5, 4.5]
    rng = [xrange, yrange]
    X = np.linspace(rng[0][0], rng[0][1], bin_count)
    Y = np.linspace(rng[1][0], rng[1][1], bin_count)
    XY = np.array(np.meshgrid(X, Y))
    kwparams = {
        "bin_count": bin_count,
        "xrange": xrange,
        "yrange": yrange,
    }

    include_tregions = include_transition_regions
    outline = get_boundary_outline(
        model,
        border_thickness=border_thickness,
        include_transition_regions=include_tregions,
        threshold=threshold,
        **kwparams
    )
    _plt.pcolor(XY[0, :, :], XY[1, :, :], outline, cmap="gray", alpha=alpha)


plt.apply_boundary = apply_boundary


def plot_shape_dataset(self, model, *args, **kwargs):
    """# Plot Shape Dataset"""
    if self.labels is None:
        plt.scatter2d(model.encoder(self.dataset), *args, **kwargs)
    else:
        plt.scatter2d(model.encoder(self.dataset), self.labels, *args, **kwargs)


ShapeDataset.plot = lambda model, *args, **kwargs: plot_shape_dataset(
    model, *args, **kwargs
)
