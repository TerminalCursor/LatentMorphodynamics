from .adversarial_distribution import *


class XGaussianDistribution(AdversarialDistribution):
    """# X Gaussian Distribution"""

    def __init__(self, N):
        """# X Gaussian Distribution Initialization

        Initializes the name and dimensions"""
        super().__init__(N)
        assert self.dims == 2, "This Distribution only Supports 2-Dimensions"
        self.full_name = "2-Dimensional X-Gaussian Distribution"
        self.name = "XG"

    def __call__(self, *args):
        """# Magic method when calling the distribution

        This method is going to be called when you use xgauss(case_count)"""
        import numpy as np

        assert len(args) == 1, "Only 1 argument supported"
        N = args[0]
        sample_base = np.zeros((4 * N, 2))
        sample_base[0 * N : (0 + 1) * N, :] = np.random.multivariate_normal(
            mean=[1, 1], cov=[[1, 0.7], [0.7, 1]], size=[N]
        )
        sample_base[1 * N : (1 + 1) * N, :] = np.random.multivariate_normal(
            mean=[-1, -1], cov=[[1, 0.7], [0.7, 1]], size=[N]
        )
        sample_base[2 * N : (2 + 1) * N, :] = np.random.multivariate_normal(
            mean=[-1, 1], cov=[[1, -0.7], [-0.7, 1]], size=[N]
        )
        sample_base[3 * N : (3 + 1) * N, :] = np.random.multivariate_normal(
            mean=[1, -1], cov=[[1, -0.7], [-0.7, 1]], size=[N]
        )
        np.random.shuffle(sample_base)
        return sample_base[:N, :]
