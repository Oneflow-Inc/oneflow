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
import os
import sys
import re
import unittest
import time
import random

import numpy as np

import oneflow as flow
import oneflow.nn as nn
import oneflow.unittest

import resnet50_model_dtr


def sync():
    flow.comm.barrier()
    # sync_tensor.numpy()


class TestDTRCorrectness(flow.unittest.TestCase):
    def setUp(self):
        super().setUp()
        assert (
            os.getenv("ONEFLOW_DISABLE_VIEW") is not None
        ), "Please set ONEFLOW_DISABLE_VIEW to True, 1 or ON"
        # wait for all previous operations to finish and
        # check the memory is empty at the beginning of every test case
        flow.comm.barrier()
        self.assertEqual(flow._oneflow_internal.dtr.allocated_memory(), 0)

    def test_dtr_correctness(test_case):
        seed = 20
        flow.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        flow.enable_dtr(True, "2500mb", 0, "eq")
        bs = 80
        model = resnet50_model_dtr.resnet50()
        model.load_state_dict(flow.load("rn50_weights"))

        criterion = nn.CrossEntropyLoss()

        cuda0 = flow.device("cuda:0")

        # enable module to use cuda
        model.to(cuda0)
        criterion.to(cuda0)

        learning_rate = 1e-3
        optimizer = flow.optim.SGD(model.parameters(), lr=learning_rate, momentum=0)

        train_data = flow.tensor(
            np.random.uniform(size=(bs, 3, 224, 224)).astype(np.float32), device=cuda0
        )
        train_label = flow.tensor(
            (np.random.uniform(size=(bs,)) * 1000).astype(np.int32),
            dtype=flow.int32,
            device=cuda0,
        )

        WARMUP_ITERS = 5
        ALL_ITERS = 40
        total_time = 0
        for iter in range(ALL_ITERS):
            for x in model.parameters():
                x.grad = flow.zeros_like(x).to(cuda0)

            if iter >= WARMUP_ITERS:
                start_time = time.time()
            logits = model(train_data)
            loss = criterion(logits, train_label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(True)
            if iter == ALL_ITERS - 1:
                print(f"loss: {loss}")
                test_case.assertGreater(loss, 5.98)
                test_case.assertLess(loss, 6.01)
            del logits
            del loss
            sync()
            if iter >= WARMUP_ITERS:
                end_time = time.time()
                this_time = end_time - start_time
                total_time += this_time

        end_time = time.time()
        print(
            f"{ALL_ITERS - WARMUP_ITERS} iters: avg {(total_time) / (ALL_ITERS - WARMUP_ITERS)}s"
        )


if __name__ == "__main__":
    unittest.main()
