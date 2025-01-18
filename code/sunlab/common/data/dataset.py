from .dataset_iterator import DatasetIterator


class Dataset:
    """# Dataset Superclass"""

    base_scale = 10.0

    def __init__(
        self,
        dataset_filename,
        data_columns=[],
        label_columns=[],
        batch_size=None,
        shuffle=False,
        val_split=0.0,
        scaler=None,
        sort_columns=None,
        random_seed=4332,
        pre_scale=10.0,
        **kwargs
    ):
        """# Initialize Dataset
        self.dataset = dataset (N, ...)
        self.labels = labels (N, ...)

        Optional Arguments:
        - prescale_function: The function that takes the ratio and transforms
        the dataset by multiplying the prescale_function output
        - sort_columns: The columns to sort the data by initially
        - equal_split: If the classifications should be equally split in
        training"""
        from pandas import read_csv
        from numpy import array, all
        from numpy.random import seed

        if seed is not None:
            seed(random_seed)

        # Basic Dataset Information
        self.data_columns = data_columns
        self.label_columns = label_columns
        self.source = dataset_filename
        self.dataframe = read_csv(self.source)

        # Pre-scaling Transformation
        prescale_ratio = self.base_scale / pre_scale
        ratio = prescale_ratio
        prescale_powers = array([2, 1, 1, 0, 2, 1, 0, 0, 1, 1, 1, 1, 1])
        if "prescale_function" in kwargs.keys():
            prescale_function = kwargs["prescale_function"]
        else:

            def prescale_function(x):
                return x**prescale_powers

        self.prescale_function = prescale_function
        self.prescale_factor = self.prescale_function(ratio)
        assert (
            len(data_columns) == self.prescale_factor.shape[0]
        ), "Column Mismatch on Prescale"
        self.original_scale = pre_scale

        # Scaling Transformation
        self.scaled = scaler is not None
        self.scaler = scaler

        # Training Dataset Information
        self.do_split = False if val_split == 0.0 else True
        self.validation_split = val_split
        self.batch_size = batch_size
        self.do_shuffle = shuffle
        self.equal_split = False
        if "equal_split" in kwargs.keys():
            self.equal_split = kwargs["equal_split"]

        # Classification Labels if they exist
        self.dataset = self.dataframe[self.data_columns].to_numpy()
        if len(self.label_columns) == 0:
            self.labels = None
        elif not all([column in self.dataframe.columns for column in label_columns]):
            import warnings

            warnings.warn(
                "No classification labels found for the dataset", RuntimeWarning
            )
            self.labels = None
        else:
            self.labels = self.dataframe[self.label_columns].squeeze()

        # Initialize the dataset
        if "sort_columns" in kwargs.keys():
            self.sort(kwargs["sort_columns"])
        if self.do_shuffle:
            self.shuffle()
        if self.do_split:
            self.split()
        self.refresh_dataset()

    def __len__(self):
        """# Get how many cases are in the dataset"""
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        """# Make Dataset Sliceable"""
        idx_slice = None
        slice_stride = 1 if self.batch_size is None else self.batch_size
        # If we pass a slice, return the slice
        if type(idx) == slice:
            idx_slice = idx
        # If we pass an int, return a batch-size slice
        else:
            idx_slice = slice(
                idx * slice_stride, min([len(self), (idx + 1) * slice_stride])
            )
        if self.labels is None:
            return self.dataset[idx_slice, ...]
        return self.dataset[idx_slice, ...], self.labels[idx_slice, ...]

    def scale_data(self, data):
        """# Scale dataset from scaling function"""
        data = data * self.prescale_factor
        if not (self.scaler is None):
            data = self.scaler(data)
        return data

    def scale(self):
        """# Scale Dataset"""
        self.dataset = self.scale_data(self.dataset)

    def refresh_dataset(self, dataframe=None):
        """# Refresh Dataset

        Regenerate the dataset from a dataframe.
        Primarily used after a sort or filter."""
        if dataframe is None:
            dataframe = self.dataframe
        self.dataset = dataframe[self.data_columns].to_numpy()
        if self.labels is not None:
            self.labels = dataframe[self.label_columns].to_numpy().squeeze()
        self.scale()

    def sort_on(self, columns):
        """# Sort Dataset on Column(s)"""
        from numpy import all

        if type(columns) == str:
            columns = [columns]
        if columns is not None:
            assert all(
                [column in self.dataframe.columns for column in columns]
            ), "Dataframe does not contain some provided columns!"
            self.dataframe = self.dataframe.sort_values(by=columns)
            self.refresh_dataset()

    def filter_on(self, column, value):
        """# Filter Dataset on Column Value(s)"""
        assert column in self.dataframe.columns, "Column DNE"
        self.working_dataset = self.dataframe[self.dataframe[column].isin(value)]
        self.refresh_dataset(self.working_dataset)

    def filter_off(self):
        """# Remove any filter on the dataset"""
        self.refresh_dataset()

    def unique(self, column):
        """# Get unique values in a column(s)"""
        assert column in self.dataframe.columns, "Column DNE"
        from numpy import unique

        return unique(self.dataframe[column])

    def shuffle_data(self, data, labels=None):
        """# Shuffle a dataset"""
        from numpy.random import permutation

        shuffled = permutation(data.shape[0])
        if labels is not None:
            assert (
                self.labels.shape[0] == self.dataset.shape[0]
            ), "Dataset and Label Shape Mismatch"
            shuf_data = data[shuffled, ...]
            shuf_labels = labels[shuffled]
            if len(labels.shape) > 1:
                shuf_labels = labels[shuffled,...]
            return shuf_data, shuf_labels
        return data[shuffled, ...]

    def shuffle(self):
        """# Shuffle the dataset"""
        if self.do_shuffle:
            if self.labels is None:
                self.dataset = self.shuffle_data(self.dataset)
            self.dataset, self.labels = self.shuffle_data(self.dataset, self.labels)

    def split(self):
        """# Training/ Validation Splitting"""
        from numpy import floor, unique, where, hstack, delete
        from numpy.random import permutation

        equal_classes = self.equal_split
        if not self.do_split:
            return
        assert self.validation_split <= 1.0, "Too High"
        assert self.validation_split > 0.0, "Too Low"
        train_count = int(floor(self.dataset.shape[0] * (1 - self.validation_split)))
        training_data = self.dataset[:train_count, ...]
        training_labels = None
        validation_data = self.dataset[train_count:, ...]
        validation_labels = None
        if self.labels is not None:
            if equal_classes:
                # Ensure the split balances the prevalence of each class
                assert len(self.labels.shape) == 1, "1D Classification Only Currently"
                classification_breakdown, classification_breakdown_counts = unique(self.labels, return_counts=True)
                train_count = min(
                    [
                        train_count,
                        classification_breakdown.shape[0]
                        * min(classification_breakdown_counts),
                    ]
                )
                class_size = int(floor(train_count / classification_breakdown.shape[0]))
                class_indicies = [
                    permutation(where(self.labels == _class)[0])
                    for _class in classification_breakdown
                ]
                tclass_indicies = [indexes[:class_size] for indexes in class_indicies]
                vclass_indicies = [indexes[class_size:] for indexes in class_indicies]
                train_class_indicies = hstack(tclass_indicies).squeeze()
                train_class_indicies = permutation(train_class_indicies)
                val_class_indicies = hstack(vclass_indicies).squeeze()
                val_class_indicies = permutation(val_class_indicies)
                training_data = self.dataset[train_class_indicies, ...]
                training_labels = self.labels[train_class_indicies]
                if len(self.labels.shape) > 1:
                    training_labels = self.labels[train_class_indicies,...]
                validation_data = self.dataset[val_class_indicies, ...]
                validation_labels = self.labels[val_class_indicies]
                if len(self.labels.shape) > 1:
                    validation_labels = self.labels[val_class_indicies,...]
            else:
                training_labels = self.labels[:train_count]
                if len(training_labels.shape) > 1:
                    training_labels = self.labels[:train_count, ...]
                validation_labels = self.labels[train_count:]
                if len(validation_labels.shape) > 1:
                    validation_labels = self.labels[train_count:, ...]
        self.training_data = training_data
        self.validation_data = validation_data
        self.training = DatasetIterator(training_data, training_labels, self.batch_size)
        self.validation = DatasetIterator(
            validation_data, validation_labels, self.batch_size
        )

    def reset_iterators(self):
        """# Reset Train/ Validation Iterators"""
        self.split()
