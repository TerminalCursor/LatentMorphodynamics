class AdversarialScaler:
    """# Scaler Class to use in Adversarial Autoencoder

    For any scaler to be implemented, make sure to ensure each of the methods
    are implemented:
    - __init__ (optional)
    - init
    - load
    - save
    - __call__"""

    def __init__(self, base_directory):
        """# Scaler initialization

        - Initialize the base directory of the model where it will live"""
        self.base_directory = base_directory

    def init(self, data):
        """# Scaler initialization

        Initialize the scaler transformation with the data
        Should always return self in subclasses"""
        raise NotImplementedError("Scaler initialization has not been implemented yet")

    def load(self):
        """# Scaler loading

        Load the data scaler model from a file
        Should always return self in subclasses"""
        raise NotImplementedError("Scaler loading has not been implemented yet")

    def save(self):
        """# Scaler saving

        Save the data scaler model"""
        raise NotImplementedError("Scaler saving has not been implemented yet")

    def transform(self, *args, **kwargs):
        """# Scale the given data"""
        return self.__call__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        """# Scale the given data"""
        raise NotImplementedError("Scaler has not been implemented yet")
