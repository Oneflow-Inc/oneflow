"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
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


if __name__ == "__main__":
    dataset = ScpDataset()
    dataloader = Data.DataLoader(dataset, batch_size=32, shuffle=True)
    for X in dataloader:
        print(X.shape)
