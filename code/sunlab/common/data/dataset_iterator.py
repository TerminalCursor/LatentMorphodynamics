class DatasetIterator:
    """# Dataset Iterator

    Creates an iterator object on a dataset and labels"""

    def __init__(self, dataset, labels=None, batch_size=None):
        """# Initialize the iterator with the dataset and labels

        - batch_size: How many to include in the iteration"""
        self.dataset = dataset
        self.labels = labels
        self.current = 0
        self.batch_size = (
            batch_size if batch_size is not None else self.dataset.shape[0]
        )

    def __iter__(self):
        """# Iterator Function"""
        return self

    def __next__(self):
        """# Next Iteration

        Slice the dataset and labels to provide"""
        self.cur = self.current
        self.current += 1
        if self.cur * self.batch_size < self.dataset.shape[0]:
            iterator_slice = slice(
                self.cur * self.batch_size, (self.cur + 1) * self.batch_size
            )
            if self.labels is None:
                return self.dataset[iterator_slice, ...]
            return self.dataset[iterator_slice, ...], self.labels[iterator_slice, ...]
        raise StopIteration
