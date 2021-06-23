import oneflow.experimental as flow
import oneflow.python.utils.data as Data

import numpy as np

flow.enable_eager_execution()


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


if __name__ == '__main__':
    dataset = ScpDataset()
    dataloader = Data.DataLoader(dataset, batch_size=32, shuffle=True)
    for X in dataloader:
        print(X.shape)

