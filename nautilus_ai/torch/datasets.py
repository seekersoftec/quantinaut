import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    """
    A PyTorch Dataset for creating sliding window sequences of input (`xs`) and target (`ys`) data.

    Parameters
    ----------
    xs : torch.Tensor
        The input data tensor of shape `(num_samples, num_features)`.
    ys : torch.Tensor
        The target data tensor of shape `(num_samples, num_targets)`.
    window_size : int
        The size of the sliding window for input sequences.

    Notes
    -----
    - The dataset generates overlapping windows from `xs` with a corresponding target from `ys`.
    - The `window_x` contains data from `index` to `index + window_size - 1`.
    - The `window_y` is the target corresponding to the last row in `window_x`.
    - Indexing is reversed to generate windows from the end of the dataset towards the start.

    Raises
    ------
    ValueError
        If `window_size` is greater than the number of samples in `xs` or if `xs` and `ys` have mismatched dimensions.
    """

    def __init__(self, xs: torch.Tensor, ys: torch.Tensor, window_size: int):
        if window_size > xs.shape[0]:
            raise ValueError(
                "`window_size` must not exceed the number of samples in `xs`."
            )
        if xs.shape[0] != ys.shape[0]:
            raise ValueError("`xs` and `ys` must have the same number of samples.")
        self.xs = xs
        self.ys = ys
        self.window_size = window_size

    def __len__(self) -> int:
        """
        Returns the total number of windows in the dataset.

        Returns
        -------
        int
            The number of windows available.
        """
        return len(self.xs) - self.window_size

    def __getitem__(self, index: int) -> tuple:
        """
        Retrieves the `index`-th window of input and target data.

        Parameters
        ----------
        index : int
            The index of the desired window.

        Returns
        -------
        tuple
            A tuple `(window_x, window_y)` where:
            - `window_x` is a tensor of shape `(window_size, num_features)`.
            - `window_y` is a tensor of shape `(1, num_targets)`.

        Notes
        -----
        Indexing is reversed to generate windows starting from the end of the dataset.
        """
        idx_rev = len(self.xs) - self.window_size - index - 1
        window_x = self.xs[idx_rev : idx_rev + self.window_size, :]
        window_y = self.ys[idx_rev + self.window_size - 1, :].unsqueeze(0)
        return window_x, window_y
