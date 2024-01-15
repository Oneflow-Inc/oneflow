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
import warnings


def symbolic_opset11():
    warnings.warn(
        "The oneflow.onnx.symbolic_opset11 interface is just to align the torch.onnx.symbolic_opset11 interface and has no practical significance."
    )


def register_custom_op_symbolic(*args, **kwargs):
    warnings.warn(
        "The oneflow.onnx.register_custom_op_symbolic interface is just to align the torch.onnx.register_custom_op_symbolic interface and has no practical significance."
    )
