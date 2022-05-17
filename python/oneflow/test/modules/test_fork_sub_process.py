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

from multiprocessing.pool import Pool
import numpy as np

from oneflow.test_utils.automated_test_util import *
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow


def _test_fork_sub_process(id):
    print("\nchild process:%s start! process id: %d" % (id, os.getpid()))
    import oneflow as flow

    x = flow.tensor(np.ones((4, 16)), device="cpu")
    y = flow.tensor(np.ones((16)), device="cpu")
    z = x + y
    assert np.array_equal(z.numpy(), np.ones((4, 16)) * 2)
    print("%s child process done! process id: %d." % (id, os.getpid()))


@flow.unittest.skip_unless_1n1d()
class TestForkSubProcess(flow.unittest.TestCase):
    def test_fork_sub_process(test_case):
        flow._oneflow_internal.eager.Sync()
        print("=============main process start=============")
        # process pool
        num_process = 4
        p = Pool(num_process)
        async_res = []
        for i in range(num_process):  # create n child processes
            # put it to pool
            async_res.append(p.apply_async(_test_fork_sub_process, args=(i,)))
        p.close()
        p.join()
        for i in range(num_process):
            test_case.assertTrue(async_res[i].successful())

        print("=============main process done!=============")


if __name__ == "__main__":
    unittest.main()
