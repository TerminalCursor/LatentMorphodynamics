from matplotlib import pyplot as plt
from sunlab.common.data.shape_dataset import ShapeDataset
from sunlab.globals import DIR_ROOT


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
    raise NotImplemented("Not Yet Implemented for PyTorch!")


plt.apply_boundary = apply_boundary
