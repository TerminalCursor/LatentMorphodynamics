class Discriminator:
    """# Discriminator Model

    Constructs a discriminator model with a certain depth of intermediate
    layers of fixed size"""

    def __init__(self, model_base_directory):
        """# Discriminator Model Initialization

        - model_base_directory: The base folder directory where the model will
        be saved/ loaded"""
        self.model_base_directory = model_base_directory

    def init(self):
        """# Initialize a new Discriminator

        Expects a model parameters file to already exist in the initialization
        base directory when initializing the model"""
        from tensorflow import keras
        from tensorflow.keras import layers

        self.load_parameters()
        assert self.depth >= 0, "Depth must be non-negative"
        self.model = keras.models.Sequential()
        if self.depth == 0:
            self.model.add(
                layers.Dense(
                    1,
                    input_shape=(self.latent_size,),
                    activation=None,
                    name="discriminator_output_vector",
                )
            )
        else:
            self.model.add(
                layers.Dense(
                    self.layer_size,
                    input_shape=(self.latent_size,),
                    activation=None,
                    name="discriminator_dense_1",
                )
            )
            if self.use_leaky_relu:
                self.model.add(layers.LeakyReLU())
            else:
                self.model.add(layers.ReLU())
            if self.dropout > 0.0:
                self.model.add(layers.Dropout(self.dropout))
            for _d in range(1, self.depth):
                self.model.add(
                    layers.Dense(
                        self.layer_size,
                        activation=None,
                        name=f"discriminator_dense_{_d+1}",
                    )
                )
                if self.use_leaky_relu:
                    self.model.add(layers.LeakyReLU())
                else:
                    self.model.add(layers.ReLU())
                if self.dropout > 0.0:
                    self.model.add(layers.Dropout(self.dropout))
            self.model.add(
                layers.Dense(
                    1, activation="sigmoid", name="discriminator_output_vector"
                )
            )
        self.model._name = "Discriminator"
        return self

    def load(self):
        """# Load an existing Discriminator"""
        from os import listdir

        if "discriminator.keras" not in listdir(
            f"{self.model_base_directory}/portable/"
        ):
            return None
        import tensorflow as tf

        self.model = tf.keras.models.load_model(
            f"{self.model_base_directory}/portable/discriminator.keras", compile=False
        )
        self.model._name = "Discriminator"
        return self

    def save(self, overwrite=False):
        """# Save the current Discriminator

        - Overwrite: overwrite any existing discriminator that has been
        saved"""
        from os import listdir

        if overwrite:
            self.model.save(f"{self.model_base_directory}/portable/discriminator.keras")
            return True
        if "discriminator.keras" in listdir(f"{self.model_base_directory}/portable/"):
            return False
        self.model.save(f"{self.model_base_directory}/portable/discriminator.keras")
        return True

    def load_parameters(self):
        """# Load Discriminator Model Parameters from File
        The file needs to have the following parameters defined:
         - data_size: int
         - adversary_layer_size: int
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
        self.layer_size = parameters["adversary_layer_size"]
        self.latent_size = parameters["latent_size"]
        self.depth = parameters["autoencoder_depth"]
        self.dropout = parameters["dropout"]
        self.use_leaky_relu = parameters["use_leaky_relu"]

    def summary(self):
        """# Returns the summary of the Discriminator model"""
        return self.model.summary()

    def __call__(self, *args, **kwargs):
        """# Callable

        When calling the discriminator class, return the model's output"""
        return self.model(*args, **kwargs)
