class Encoder:
    """# Encoder Model

    Constructs an encoder model with a certain depth of intermediate layers of
    fixed size"""

    def __init__(self, model_base_directory):
        """# Encoder Model Initialization

        - model_base_directory: The base folder directory where the model will
        be saved/ loaded"""
        self.model_base_directory = model_base_directory

    def init(self):
        """# Initialize a new Encoder

        Expects a model parameters file to already exist in the initialization
        base directory when initializing the model"""
        from tensorflow import keras
        from tensorflow.keras import layers

        # Load in the model parameters
        self.load_parameters()
        assert self.depth >= 0, "Depth must be non-negative"

        # Create the model
        self.model = keras.models.Sequential()
        # At zero depth, connect input and output layer directly
        if self.depth == 0:
            self.model.add(
                layers.Dense(
                    self.latent_size,
                    input_shape=(self.data_size,),
                    activation=None,
                    name="encoder_latent_vector",
                )
            )
        # Otherwise, add fixed-sized layers between them
        else:
            self.model.add(
                layers.Dense(
                    self.layer_size,
                    input_shape=(self.data_size,),
                    activation=None,
                    name="encoder_dense_1",
                )
            )
            # Use LeakyReLU if specified
            if self.use_leaky_relu:
                self.model.add(layers.LeakyReLU())
            else:
                self.model.add(layers.ReLU())
            # Include a droput layer if specified
            if self.dropout > 0.0:
                self.model.add(layers.Dropout(self.dropout))
            for _d in range(1, self.depth):
                self.model.add(
                    layers.Dense(
                        self.layer_size, activation=None, name=f"encoder_dense_{_d+1}"
                    )
                )
                # Use LeakyReLU if specified
                if self.use_leaky_relu:
                    self.model.add(layers.LeakyReLU())
                else:
                    self.model.add(layers.ReLU())
                # Include a droput layer if specified
                if self.dropout > 0.0:
                    self.model.add(layers.Dropout(self.dropout))
            self.model.add(
                layers.Dense(
                    self.latent_size, activation=None, name="encoder_latent_vector"
                )
            )
        self.model._name = "Encoder"
        return self

    def load(self):
        """# Load an existing Encoder"""
        from os import listdir

        # If the encoder is not found, return None
        if "encoder.keras" not in listdir(f"{self.model_base_directory}/portable/"):
            return None
        # Otherwise, load the encoder
        #  compile=False suppresses warnings about training
        #  If you want to train it, you will need to recompile it
        import tensorflow as tf

        self.model = tf.keras.models.load_model(
            f"{self.model_base_directory}/portable/encoder.keras", compile=False
        )
        self.model._name = "Encoder"
        return self

    def save(self, overwrite=False):
        """# Save the current Encoder

        - Overwrite: overwrite any existing encoder that has been saved"""
        from os import listdir

        if overwrite:
            self.model.save(f"{self.model_base_directory}/portable/encoder.keras")
            return True
        if "encoder.keras" in listdir(f"{self.model_base_directory}/portable/"):
            return False
        self.model.save(f"{self.model_base_directory}/portable/encoder.keras")
        return True

    def load_parameters(self):
        """# Load Encoder Model Parameters from File
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
        """# Returns the summary of the Encoder model"""
        return self.model.summary()

    def __call__(self, *args, **kwargs):
        """# Callable

        When calling the encoder class, return the model's output"""
        return self.model(*args, **kwargs)
