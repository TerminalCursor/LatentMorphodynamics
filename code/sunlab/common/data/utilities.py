from .shape_dataset import ShapeDataset
from ..scaler.max_abs_scaler import MaxAbsScaler


def import_10x(
    filename,
    magnification=10,
    batch_size=None,
    shuffle=False,
    val_split=0.0,
    scaler=None,
    sort_columns=None,
):
    """# Import a 10x Image Dataset
    
    Pixel-to-micron: ???"""
    magnification = 10
    dataset = ShapeDataset(
        filename,
        batch_size=batch_size,
        shuffle=shuffle,
        pre_scale=magnification,
        val_split=val_split,
        scaler=scaler,
        sort_columns=sort_columns,
    )
    return dataset


def import_20x(
    filename,
    magnification=10,
    batch_size=None,
    shuffle=False,
    val_split=0.0,
    scaler=None,
    sort_columns=None,
):
    """# Import a 20x Image Dataset
    
    Pixel-to-micron: ???"""
    magnification = 20
    dataset = ShapeDataset(
        filename,
        batch_size=batch_size,
        shuffle=shuffle,
        pre_scale=magnification,
        val_split=val_split,
        scaler=scaler,
        sort_columns=sort_columns,
    )
    return dataset


def import_dataset(
    filename,
    magnification,
    batch_size=None,
    shuffle=False,
    val_split=0.0,
    scaler=None,
    sort_columns=None,
):
    """# Import a dataset

    Requires a magnificaiton to be specified"""
    dataset = ShapeDataset(
        filename,
        pre_scale=magnification,
        batch_size=batch_size,
        shuffle=shuffle,
        val_split=val_split,
        scaler=scaler,
        sort_columns=sort_columns,
    )
    return dataset


def import_full_dataset(fname, magnification=20, scaler=None):
    """# Import a Full Dataset

    If a classification file exists(.txt with a 'Class' header and 'frame','cellnumber' headers), also import it"""
    from os.path import isfile
    import pandas as pd
    import numpy as np

    cfname = fname
    tfname = cfname[:-3] + "txt"
    columns = [
        "frame",
        "cellnumber",
        "x-cent",
        "y-cent",
        "actinedge",
        "filopodia",
        "bleb",
        "lamellipodia",
    ]
    if isfile(tfname):
        dataset = import_dataset(cfname, magnification=magnification, scaler=scaler)
        class_df = np.loadtxt(tfname, skiprows=1)
        class_df = pd.DataFrame(class_df, columns=columns)
        full_df = pd.merge(
            dataset.dataframe,
            class_df,
            left_on=["Frames", "CellNum"],
            right_on=["frame", "cellnumber"],
        )
        full_df["Class"] = np.argmax(
            class_df[["actinedge", "filopodia", "bleb", "lamellipodia"]].to_numpy(),
            axis=-1,
        )
        dataset.labels = full_df["Class"].to_numpy()
    else:
        dataset = import_dataset(cfname, magnification=magnification, scaler=scaler)
        full_df = dataset.dataframe
    dataset.dataframe = full_df
    dataset.filter_off()
    return dataset
