from .adversarial_scaler import AdversarialScaler


class MaxAbsScaler(AdversarialScaler):
    """# MaxAbsScaler

    Scale the data based on the maximum-absolute value in each column"""

    def __init__(self, base_directory):
        """# MaxAbsScaler initialization

        - Initialize the base directory of the model where it will live
        - Initialize the scaler model"""
        super().__init__(base_directory)
        from sklearn.preprocessing import MaxAbsScaler as MAS

        self.scaler_base = MAS()
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

        with open(f"{self.base_directory}/portable/maxabs_scaler.pkl", "rb") as fhandle:
            self.scaler = load(fhandle)
        return self

    def save(self):
        """# Scaler saving

        Save the data scaler model"""
        from pickle import dump

        with open(f"{self.base_directory}/portable/maxabs_scaler.pkl", "wb") as fhandle:
            dump(self.scaler, fhandle)

    def __call__(self, *args, **kwargs):
        """# Scale the given data"""
        return self.scaler.transform(*args, **kwargs)
