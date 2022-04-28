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
import numpy as np
import oneflow as flow

numpy_dtype_to_oneflow_dtype_dict = {
    np.int32: flow.int32,
    np.int64: flow.int64,
    np.int8: flow.int8,
    np.uint8: flow.uint8,
    np.bool: flow.bool,
    np.float64: flow.float64,
    np.float32: flow.float32,
    np.float16: flow.float16,
}


def as_tensor(data, dtype=None, device=None):
    if flow.is_tensor(data):
        if dtype is None:
            dtype = data.dtype
        if device is None:
            device = data.device
        if data.dtype is dtype and data.device is device:
            return data
        else:
            data = data.to(dtype=dtype, device=device)
    elif isinstance(data, (np.ndarray)):
        if dtype is None:
            if (device is None) or (device.type == "cpu"):
                data = flow.from_numpy(data)
            else:
                data = flow.tensor(data, device=device)
        else:
            if data.dtype in numpy_dtype_to_oneflow_dtype_dict:
                data_infer_flow_type = numpy_dtype_to_oneflow_dtype_dict[data.dtype]
            else:
                raise TypeError("numpy-ndarray holds elements of unsupported datatype")
            if data_infer_flow_type is dtype:
                if (device is None) or (device.type == "cpu"):
                    data = flow.from_numpy(data)
                else:
                    data = flow.tensor(data, dtype=dtype, device=device)
            else:
                if (device is None) or (device.type == "cpu"):
                    data = flow.tensor(data, dtype=dtype)
                else:
                    data = flow.tensor(data, dtype=dtype, device=device)
    else:
        # handle tuple, list, scalar
        data = np.array(data)
        # not shared memory in this case
        data = flow.tensor(data)
        if device is not None:
            data = data.to(device)
        if dtype is not None:
            data = data.to(dtype)
    return data
