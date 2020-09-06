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


class PadMixin(object):
    @classmethod
    def get_padding_as_op(cls, x, pads):
        num_dim = int(len(pads) / 2)

        flow_pads = (
            np.transpose(np.array(pads).reshape([2, num_dim])).astype(np.int32).tolist()
        )
        # flow_pads = [0, 0, 0, 0] + flow_pads.flatten().tolist()
        flow_pads = [(0, 0), (0, 0)] + flow_pads

        return flow.pad(x, flow_pads)
