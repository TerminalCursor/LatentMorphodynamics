# Higher-level functions

from sunlab.common.distribution.adversarial_distribution import AdversarialDistribution
from sunlab.common.scaler.adversarial_scaler import AdversarialScaler
from sunlab.common.data.utilities import import_dataset
from .adversarial_autoencoder import AdversarialAutoencoder


def create_aae(
    dataset_file_name,
    model_directory,
    normalization_scaler: AdversarialScaler,
    distribution: AdversarialDistribution or None,
    magnification=10,
    latent_size=2,
):
    """# Create Adversarial Autoencoder

    - dataset_file_name: str = Path to the dataset file
    - model_directory: str = Path to save the model in
    - normalization_scaler: AdversarialScaler = Data normalization Scaler Model
    - distribution: AdversarialDistribution = Distribution for the Adversary
    - magnification: int = The Magnification of the Dataset"""
    dataset = import_dataset(dataset_file_name, magnification)
    model = AdversarialAutoencoder(
        model_directory, distribution, normalization_scaler
    ).init(dataset.dataset, latent_size=latent_size)
    return model


def create_aae_and_dataset(
    dataset_file_name,
    model_directory,
    normalization_scaler: AdversarialScaler,
    distribution: AdversarialDistribution or None,
    magnification=10,
    batch_size=1024,
    shuffle=True,
    val_split=0.1,
    latent_size=2,
):
    """# Create Adversarial Autoencoder and Load the Dataset

    - dataset_file_name: str = Path to the dataset file
    - model_directory: str = Path to save the model in
    - normalization_scaler: AdversarialScaler = Data normalization Scaler Model
    - distribution: AdversarialDistribution = Distribution for the Adversary
    - magnification: int = The Magnification of the Dataset"""
    model = create_aae(
        dataset_file_name,
        model_directory,
        normalization_scaler,
        distribution,
        magnification=magnification,
        latent_size=latent_size,
    )
    dataset = import_dataset(
        dataset_file_name,
        magnification,
        batch_size=batch_size,
        shuffle=shuffle,
        val_split=val_split,
        scaler=model.scaler,
    )
    return model, dataset


def load_aae(model_directory, normalization_scaler: AdversarialScaler):
    """# Load Adversarial Autoencoder

    - model_directory: str = Path to save the model in
    - normalization_scaler: AdversarialScaler = Data normalization Scaler Model
    """
    return AdversarialAutoencoder(model_directory, None, normalization_scaler).load()


def load_aae_and_dataset(
    dataset_file_name,
    model_directory,
    normalization_scaler: AdversarialScaler,
    magnification=10,
):
    """# Load Adversarial Autoencoder

    - dataset_file_name: str = Path to the dataset file
    - model_directory: str = Path to save the model in
    - normalization_scaler: AdversarialScaler = Data normalization Scaler Model
    - magnification: int = The Magnification of the Dataset"""
    model = load_aae(model_directory, normalization_scaler)
    dataset = import_dataset(
        dataset_file_name, magnification=magnification, scaler=model.scaler
    )
    return model, dataset
