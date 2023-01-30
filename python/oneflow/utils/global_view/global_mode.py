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
import oneflow._oneflow_internal.global_view as internal_global_view


class global_mode(internal_global_view.global_mode):
    r"""Create a scope to provide global information for the computation process within it.
    
    It provides convinience for converting from local execution to global execution, especially for converting to ddp global execution.
    
    1) Make the source op create the global tensor directly.
    2) Make it legal for the "to(device)" API  of the global tensor.
    3) Make it legal to use ".device" to get the device type of the global tensor.
    
    Note:
        Both placement and sbp are required if the global mode is enabled.
        
    Args:
        enabled (bool): whether the global mode is enbaled.
        placement (oneflow.placement, optional): the desired placement of the input. Default: None
        sbp (oneflow.sbp.sbp, list/tuple of oneflow.sbp.sbp, optional): the desired sbp of the input or self-defined functions in order to specify SBP. Default: None

    For example:

    .. code-block:: python

        class LinearEvalGraphWithDDP(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.linear_dp = linear_dp

            def build(self, x):
                with global_mode(True, placement=P, sbp=B):
                    device = self.linear_dp.weight.device

                    x = x.to(device)

                    out = self.linear_dp(x)

                    # The local tensor will be converted to global
                    sample = flow.randn(out.shape, device="cpu").to(device)
                    out = out + sample * 100
                    out = out - sample * 100

                return out
         
    .. code-block:: python       

        with global_mode(False):
            # The tensor will be keeped as local.
            sample = flow.randn(out.shape, device="cpu").to(device)
            out = out + sample * 100
            out = out - sample * 100
    """

    def __init__(self, enabled, placement=None, sbp=None) -> None:
        if not enabled:
            super().__init__(enabled)
        else:
            super().__init__(enabled, placement, sbp)

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        pass


class current_global_mode(internal_global_view.current_global_mode):
    r"""Get the current global mode information.
    
    Use the current_global_mode to get the information of global mode, including enabled, placement and sbp.

    Note: 
        The sbp property is supposed to return a list/tuple of `oneflow.sbp.sbp`.

    For example:

    .. code-block:: python

        with global_mode(True, placement=P, sbp=B):
            # Get the global mode info.
            cur_global_mode = global_view.current_global_mode()
            test_case.assertTrue(cur_global_mode.is_enabled)
            test_case.assertEqual(cur_global_mode.placement, P)
            test_case.assertEqual(cur_global_mode.sbp[0], B)
    """

    def __init__(self) -> None:
        super().__init__()

    @property
    def is_enabled(self):
        return super().is_enabled

    @property
    def sbp(self):
        return super().sbp

    @property
    def placement(self):
        return super().placement
