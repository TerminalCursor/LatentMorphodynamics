class AdversarialDistribution:
    """# Distribution Class to use in Adversarial Autoencoder

    For any distribution to be implemented, make sure to ensure each of the
    methods are implemented"""

    def __init__(self, N):
        """# Initialize the distribution for N-dimensions"""
        self.dims = N
        return

    def get_full_name(self):
        """# Return a human-readable name of the distribution"""
        return self.full_name

    def get_name(self):
        """# Return a shortened name of the distribution

        Preferrably, the name should include characters that can be included in
        a file name"""
        return self.name

    def __str__(self):
        """# Returns the short name"""
        return self.get_name()

    def __repr__(self):
        """# Returns the short name"""
        return self.get_name()

    def __call__(self, *args):
        """# Magic method when calling the distribution

        This method is going to be called when you use `dist(...)`"""
        raise NotImplementedError("This distribution has not been implemented yet")
