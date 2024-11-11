import numpy


numpy.load_dat = lambda *args, **kwargs: numpy.load(
    *args, **kwargs, allow_pickle=True
).item()
