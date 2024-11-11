from .adversarial_distribution import *


class GaussianDistribution(AdversarialDistribution):
    """# Gaussian Distribution"""

    def __init__(self, N):
        """# Gaussian Distribution Initialization

        Initializes the name and dimensions"""
        super().__init__(N)
        self.full_name = f"{N}-Dimensional Gaussian Distribution"
        self.name = "G"

    def __call__(self, *args):
        """# Magic method when calling the distribution

        This method is going to be called when you use gauss(N1,...,Nm)"""
        import numpy as np

        return np.random.multivariate_normal(
            mean=np.zeros(self.dims), cov=np.eye(self.dims), size=[*args]
        )
