from sunlab.common import ShapeDataset
from sunlab.common import MaxAbsScaler


def process_and_load_dataset(
    dataset_file, model_folder, magnification=10, scaler=MaxAbsScaler
):
    """# Load a dataset and process a models' Latent Space on the Dataset"""
    from ..models import load_aae
    from sunlab.common import import_full_dataset

    model = load_aae(model_folder, normalization_scaler=scaler)
    dataset = import_full_dataset(
        dataset_file, magnification=magnification, scaler=model.scaler
    )
    latent = model.encoder(dataset.dataset).numpy()
    assert len(latent.shape) == 2, "Only 1D Latent Vectors Supported"
    for dim in range(latent.shape[1]):
        dataset.dataframe[f"Latent-{dim}"] = latent[:, dim]
    return dataset


def process_and_load_datasets(
    dataset_file_list, model_folder, magnification=10, scaler=MaxAbsScaler
):
    from pandas import concat
    from ..models import load_aae

    dataframes = []
    datasets = []
    for dataset_file in dataset_file_list:
        dataset = process_and_load_dataset(
            dataset_file, model_folder, magnification, scaler
        )
        model = load_aae(model_folder, normalization_scaler=scaler)
        dataframe = dataset.dataframe
        for label in ["ActinEdge", "Filopodia", "Bleb", "Lamellipodia"]:
            if label in dataframe.columns:
                dataframe[label.lower()] = dataframe[label]
            if label.lower() not in dataframe.columns:
                dataframe[label.lower()] = 0
        latent_columns = [f"Latent-{dim}" for dim in range(model.latent_size)]
        datasets.append(dataset)
        dataframes.append(
            dataframe[
                dataset.data_columns
                + dataset.label_columns
                + latent_columns
                + ["Frames", "CellNum"]
                + ["actinedge", "filopodia", "bleb", "lamellipodia"]
            ]
        )
    return datasets, concat(dataframes)
