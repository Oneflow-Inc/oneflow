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
import oneflow_api
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("device")
class Device(oneflow_api.device):
    def __init__(self, device: str):
        available_device = ["cpu", "cuda"]
        device_list = device.split(":")
        if len(device_list) > 2:
            raise RuntimeError("Invalid device string: ", device)
        device_type = device_list[0]
        if device_type not in available_device:
            raise RuntimeError(
                "Expected one of cpu, cuda device type at start of device string "
                + device_type,
            )
        if len(device_list) > 1:
            device_index = int(device_list[1])
            if device_type == "cpu" and device_index != 0:
                raise RuntimeError("CPU device index must be 0")
        else:
            device_index = 0
        oneflow_api.device.__init__(self, device_type, device_index)
