from .adversarial_distribution import *


class UniformDistribution(AdversarialDistribution):
    """# Uniform Distribution on [0, 1)"""

    def __init__(self, N):
        """# Uniform Distribution Initialization

        Initializes the name and dimensions"""
        super().__init__(N)
        self.full_name = f"{N}-Dimensional Uniform Distribution"
        self.name = "U"

    def __call__(self, *args):
        """# Magic method when calling the distribution

        This method is going to be called when you use uniform(N1,...,Nm)"""
        import numpy as np

        return np.random.rand(*args, self.dims)
