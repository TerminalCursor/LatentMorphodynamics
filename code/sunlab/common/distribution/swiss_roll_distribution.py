from .adversarial_distribution import *


class SwissRollDistribution(AdversarialDistribution):
    """# Swiss Roll Distribution"""

    def __init__(self, N, scaling_factor=0.25, noise_level=0.15):
        """# Swiss Roll Distribution Initialization

        Initializes the name and dimensions"""
        super().__init__(N)
        assert (self.dims == 2) or (
            self.dims == 3
        ), "This Distribution only Supports 2,3-Dimensions"
        self.full_name = f"{self.dims}-Dimensional Swiss Roll Distribution Distribution"
        self.name = f"SR{self.dims}"
        self.noise_level = noise_level
        self.scale = scaling_factor

    def __call__(self, *args):
        """# Magic method when calling the distribution

        This method is going to be called when you use xgauss(case_count)"""
        import numpy as np

        assert len(args) == 1, "Only 1 argument supported"
        N = args[0]
        noise = self.noise_level
        scaling_factor = self.scale

        t = 3 * np.pi / 2 * (1 + 2 * np.random.rand(1, N))
        h = 21 * np.random.rand(1, N)
        RANDOM = np.random.randn(3, N) * noise
        data = (
            np.concatenate(
                (scaling_factor * t * np.cos(t), h, scaling_factor * t * np.sin(t))
            )
            + RANDOM
        )
        if self.dims == 2:
            return data.T[:, [0, 2]]
        return data.T[:, [0, 2, 1]]
