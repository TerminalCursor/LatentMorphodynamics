from torch.autograd import Variable

EPS = 1e-15


def to_var(x):
    """# Convert to variable"""
    import torch

    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)
