class EncoderDiscriminator:
    """# EncoderDiscriminator Model

    Constructs an encoder-discriminator model"""

    def __init__(self, model_base_directory):
        """# EncoderDiscriminator Model Initialization

        - model_base_directory: The base folder directory where the model will
        be saved/ loaded"""
        self.model_base_directory = model_base_directory

    def init(self, encoder, discriminator):
        """# Initialize a EncoderDiscriminator

        - encoder: The encoder to use
        - discriminator: The discriminator to use"""
        from tensorflow import keras

        self.load_parameters()
        self.model = keras.models.Sequential()
        self.model.add(encoder.model)
        self.model.add(discriminator.model)
        self.model._name = "EncoderDiscriminator"
        return self

    def load(self):
        """# Load an existing EncoderDiscriminator"""
        from os import listdir

        if "encoder_discriminator.keras" not in listdir(
            f"{self.model_base_directory}/portable/"
        ):
            return None
        import tensorflow as tf

        self.model = tf.keras.models.load_model(
            f"{self.model_base_directory}/portable/encoder_discriminator" + ".keras",
            compile=False,
        )
        self.model._name = "EncoderDiscriminator"
        return self

    def save(self, overwrite=False):
        """# Save the current EncoderDiscriminator

        - Overwrite: overwrite any existing encoder_discriminator that has been
        saved"""
        from os import listdir

        if overwrite:
            self.model.save(
                f"{self.model_base_directory}/portable/encoder_discriminator" + ".keras"
            )
            return True
        if "encoder_discriminator.keras" in listdir(
            f"{self.model_base_directory}/portable/"
        ):
            return False
        self.model.save(
            f"{self.model_base_directory}/portable/encoder_discriminator" + ".keras"
        )
        return True

    def load_parameters(self):
        """# Load EncoderDiscriminator Model Parameters from File
        The file needs to have the following parameters defined:
         - data_size: int
         - autoencoder_layer_size: int
         - latent_size: int
         - autoencoder_depth: int
         - dropout: float (set to 0. if you don't want a dropout layer)
         - use_leaky_relu: boolean"""
        from pickle import load

        with open(
            f"{self.model_base_directory}/portable/model_parameters.pkl", "rb"
        ) as phandle:
            parameters = load(phandle)
        self.data_size = parameters["data_size"]
        self.layer_size = parameters["autoencoder_layer_size"]
        self.latent_size = parameters["latent_size"]
        self.depth = parameters["autoencoder_depth"]
        self.dropout = parameters["dropout"]
        self.use_leaky_relu = parameters["use_leaky_relu"]

    def summary(self):
        """# Returns the summary of the EncoderDiscriminator model"""
        return self.model.summary()

    def __call__(self, *args, **kwargs):
        """# Callable

        When calling the encoder_discriminator class, return the model's
        output"""
        return self.model(*args, **kwargs)
