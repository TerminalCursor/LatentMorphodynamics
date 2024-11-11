from .adversarial_scaler import AdversarialScaler


class QuantileScaler(AdversarialScaler):
    """# QuantileScaler

    Scale the data based on the quantile distributions of each column"""

    def __init__(self, base_directory):
        """# QuantileScaler initialization

        - Initialize the base directory of the model where it will live
        - Initialize the scaler model"""
        super().__init__(base_directory)
        from sklearn.preprocessing import QuantileTransformer as QS

        self.scaler_base = QS()
        self.scaler = None

    def init(self, data):
        """# Scaler initialization

        Initialize the scaler transformation with the data"""
        self.scaler = self.scaler_base.fit(data)
        return self

    def load(self):
        """# Scaler loading

        Load the data scaler model from a file"""
        from pickle import load

        with open(
            f"{self.base_directory}/portable/quantile_scaler.pkl", "rb"
        ) as fhandle:
            self.scaler = load(fhandle)
        return self

    def save(self):
        """# Scaler saving

        Save the data scaler model"""
        from pickle import dump

        with open(
            f"{self.base_directory}/portable/quantile_scaler.pkl", "wb"
        ) as fhandle:
            dump(self.scaler, fhandle)

    def __call__(self, *args, **kwargs):
        """# Scale the given data"""
        return self.scaler.transform(*args, **kwargs)
