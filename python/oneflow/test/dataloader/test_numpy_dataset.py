import unittest

import numpy as np

import oneflow as flow
import oneflow.utils.data as Data


class ScpDataset(Data.Dataset):
    def __init__(self, chunksize=200, dim=81, length=2000):
        self.chunksize = chunksize
        self.dim = dim
        self.length = length

    def __getitem__(self, index):
        np.random.seed(index)
        return np.random.randn(self.chunksize, self.dim)

    def __len__(self):
        return self.length


@flow.unittest.skip_unless_1n1d()
@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestNumpyDataset(flow.unittest.TestCase):
    def test_numpy_dataset(test_case):
        dataset = ScpDataset()
        dataloader = Data.DataLoader(dataset, batch_size=16, shuffle=True)
        for X in dataloader:
            test_case.assertEqual(X.shape, flow.Size([16, 200, 81]))


if __name__ == "__main__":
    unittest.main()
