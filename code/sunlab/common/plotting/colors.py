class PhenotypeColors:
    """# Phenotype Colorings

    Standardization for the different phenotype colors"""

    def __init__(self):
        """# Empty Construtor"""
        pass

    def get_basic_colors(self, transition=False):
        """# Return the Color Names

        - transition: Returns the color for the transition class too"""
        if transition:
            return ["yellow", "purple", "green", "blue", "cyan"]
        return ["yellow", "purple", "green", "blue"]

    def get_colors(self, transition=False):
        """# Return the Color Names

        - transition: Returns the color for the transition class too"""
        if transition:
            return ["#ffff00", "#ff3cfa", "#11f309", "#213ff0", "cyan"]
        return ["#ffff00", "#ff3cfa", "#11fe09", "#213ff0"]

    def get_colormap(self, transition=False):
        """# Return the Matplotlib Colormap

        - transition: Returns the color for the transition class too"""
        from matplotlib.colors import ListedColormap as LC

        return LC(self.get_colors(transition))


# Basic Exports
Pcolor = PhenotypeColors().get_colors()
Pmap = PhenotypeColors().get_colormap()
Pmapx = PhenotypeColors().get_colormap(True)
