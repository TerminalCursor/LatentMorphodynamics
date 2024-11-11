class ImageDataset:
    def __init__(
        self,
        base_directory,
        ext="png",
        channels=[0],
        batch_size=None,
        shuffle=False,
        rotate=False,
        rotate_p=1.,
    ):
        """# Image Dataset

        Load a directory of images"""
        from glob import glob
        from matplotlib.pyplot import imread
        from numpy import newaxis, vstack
        from numpy.random import permutation, rand

        self.base_directory = base_directory
        files = glob(self.base_directory + "*." + ext)
        self.dataset = []
        for file in files:
            im = imread(file)[newaxis, :, :, channels].transpose(0, 3, 1, 2)
            self.dataset.append(im)
            # Also add rotations of the image to the dataset
            if rotate:
                if rand() < rotate_p:
                    self.dataset.append(im[:, :, ::-1, :])
                if rand() < rotate_p:
                    self.dataset.append(im[:, :, :, ::-1])
                if rand() < rotate_p:
                    self.dataset.append(im[:, :, ::-1, ::-1])
                if rand() < rotate_p:
                    self.dataset.append(im.transpose(0, 1, 3, 2))
                if rand() < rotate_p:
                    self.dataset.append(im.transpose(0, 1, 3, 2)[:, :, ::-1, :])
                if rand() < rotate_p:
                    self.dataset.append(im.transpose(0, 1, 3, 2)[:, :, :, ::-1])
                if rand() < rotate_p:
                    self.dataset.append(im.transpose(0, 1, 3, 2)[:, :, ::-1, ::-1])
        self.dataset = vstack(self.dataset)
        if shuffle:
            self.dataset = self.dataset[permutation(self.dataset.shape[0]), ...]
        self.batch_size = (
            batch_size if batch_size is not None else self.dataset.shape[0]
        )

    def torch(self, device=None):
        """# Cast to Torch Tensor"""
        import torch

        if device is None:
            device = torch.device("cpu")
        return torch.tensor(self.dataset).to(device)

    def numpy(self):
        """# Cast to Numpy Array"""
        return self.dataset

    def __len__(self):
        """# Return Number of Cases

        (or Number in each Batch)"""
        return self.dataset.shape[0] // self.batch_size

    def __getitem__(self, index):
        """# Slice By Batch"""
        if type(index) == tuple:
            return self.dataset[index]
        elif type(index) == int:
            return self.dataset[
                index * self.batch_size : (index + 1) * self.batch_size, ...
            ]
        return
