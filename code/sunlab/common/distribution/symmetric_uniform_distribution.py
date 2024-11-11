from .adversarial_distribution import *


class SymmetricUniformDistribution(AdversarialDistribution):
    """# Symmetric Uniform Distribution on [-1, 1)"""

    def __init__(self, N):
        """# Symmetric Uniform Distribution Initialization

        Initializes the name and dimensions"""
        super().__init__(N)
        self.full_name = f"{N}-Dimensional Symmetric Uniform Distribution"
        self.name = "SU"

    def __call__(self, *args):
        """# Magic method when calling the distribution

        This method is going to be called when you use suniform(N1,...,Nm)"""
        import numpy as np

        return np.random.rand(*args, self.dims) * 2.0 - 1.0
