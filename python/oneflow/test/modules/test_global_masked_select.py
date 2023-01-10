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
import unittest

import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *

# Not check graph because of one reason:
# Reason 1, The implementation of the masked_select op calls argwhere with the lazy tensor as an argument, but lazy tensor can not be applied to argwhere.
# Please refer to File "python/oneflow/nn/modules/masked_select.py", line 54, in masked_select_op.
@autotest(n=1, check_graph="ValidatedFalse")
def _test_masked_select(test_case, placement, sbp):
    k1 = random(1, 2).to(int).value() * 8
    k2 = random(1, 2).to(int).value() * 8
    input = random_tensor(ndim=2, dim0=k1, dim1=k2).to_global(placement, sbp)
    mask = input.ge(0.5)
    return torch.masked_select(input, mask)


# Not check graph because of one reason:
# Reason 1, The implementation of the masked_select op calls argwhere with the lazy tensor as an argument, but lazy tensor can not be applied to argwhere.
# Please refer to File "python/oneflow/nn/modules/masked_select.py", line 54, in masked_select_op.
@autotest(n=1, check_graph="ValidatedFalse")
def _test_masked_select_broadcast(test_case, placement, input_sbp, mask_sbp):
    k1 = random(1, 2).to(int).value() * 8
    k2 = random(1, 2).to(int).value() * 8
    input = random_tensor(ndim=4, dim0=k1, dim1=k2, dim2=1, dim3=k2).to_global(
        placement, input_sbp
    )
    mask = random_tensor(ndim=4, dim0=k1, dim1=k2, dim2=k1, dim3=1).to_global(
        placement, mask_sbp
    )
    return torch.masked_select(input, mask > 0.5)


class TestMaskedSelect(flow.unittest.TestCase):
    @globaltest
    def test_masked_select(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_masked_select(test_case, placement, sbp)

    @globaltest
    def test_masked_select_broadcast(test_case):
        for placement in all_placement():
            for input_sbp in all_sbp(placement, valid_split_axis=[0, 1, 3]):
                for mask_sbp in all_sbp(placement, max_dim=3):
                    _test_masked_select_broadcast(
                        test_case, placement, input_sbp, mask_sbp
                    )


if __name__ == "__main__":
    unittest.main()
