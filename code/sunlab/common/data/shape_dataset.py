from .dataset import Dataset


class ShapeDataset(Dataset):
    """# Shape Dataset"""

    def __init__(
        self,
        dataset_filename,
        data_columns=[
            "Area",
            "MjrAxisLength",
            "MnrAxisLength",
            "Eccentricity",
            "ConvexArea",
            "EquivDiameter",
            "Solidity",
            "Extent",
            "Perimeter",
            "ConvexPerim",
            "FibLen",
            "InscribeR",
            "BlebLen",
        ],
        label_columns=["Class"],
        batch_size=None,
        shuffle=False,
        val_split=0.0,
        scaler=None,
        sort_columns=None,
        random_seed=4332,
        pre_scale=10,
        **kwargs
    ):
        """# Initialize Dataset
        self.dataset = dataset (N, ...)
        self.labels = labels (N, ...)

        Optional Arguments:
        - prescale_function: The function that takes the ratio and transforms
        the dataset by multiplying the prescale_function output
        - sort_columns: The columns to sort the data by initially
        - equal_split: If the classifications should be equally split in
        training"""
        super().__init__(
            dataset_filename,
            data_columns=data_columns,
            label_columns=label_columns,
            batch_size=batch_size,
            shuffle=shuffle,
            val_split=val_split,
            scaler=scaler,
            sort_columns=sort_columns,
            random_seed=random_seed,
            pre_scale=pre_scale,
            **kwargs
        )
