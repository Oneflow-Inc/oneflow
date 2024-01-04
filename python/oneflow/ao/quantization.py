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


class DeQuantStub:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "The oneflow.ao.DeQuantStub interface is just to align the torch.ao.DeQuantStub interface and has no practical significance."
        )


class QuantStub:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "The oneflow.ao.QuantStub interface is just to align the torch.ao.QuantStub interface and has no practical significance."
        )
