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


import oneflow as flow
import unittest

from automated_test_util import *



@autotest()
def test_index_select_by_gpu(test_case):
    device = random_device()

    #test 4 dimensions tensor
    axis = radom(0,4).to(int)
    index= random_pytorch_tensor(ndim=1,low=0, high=dim[axis], dtype=int).to(device)

    dim = []
    for i in range(0,4):
        dim.append(random(2,6).to(int))

    x = random_pytorch_tensor(ndim=4, dim0=dim[0], dim1=dim[1], dim2=dim[2], dim3=dim[3]).to(device)  
    y = torch.index_select(input=x, dim=axis, index=index)  

    return y

if __name__ == "__main__":
   unittest.main() 
