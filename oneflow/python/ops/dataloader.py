
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
from __future__ import absolute_import

class DataLoader(object):
    r""" Data loader, provides an iterator over the given dataset.
    """
    def __init__(self, dataset, batch_size=1):
        self.data_set = dataset
        self.batch_size = batch_size

    def __iter__(self):
        return _DataLoaderIter(self)

class _DataLoaderIter(object):
    def __init__(self, loader):
        self._dataset = loader.dataset
    
    def __iter__(self):
        return self
    
    def _next_index(self):
        return next(self._sampler_iter)

    def _next_data(self):
        index = self._next_index()
        #data = self._dataset_fetcher.fetch(index)
        return data
    
    def __next__(self):
        data = self._next_data()
        return data
    
    def __len__(self):
        return NotImplementedError
