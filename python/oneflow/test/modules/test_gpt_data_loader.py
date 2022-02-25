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
import os
import numpy as np

import oneflow as flow
import oneflow.unittest


class GPTDataLoader(flow.nn.Module):
    def __init__(
        self,
        data_file_prefix="/dataset/Megatron-LM/dummy/gpt_sample_dataset_text_document",
        seq_length=1024,
        num_samples=648,
        batch_size=8,
        shuffle=True,
        random_seed=12345,
        device=None,
        placement=None,
        sbp=None,
    ):
        super().__init__()
        self.loader_ = flow.nn.GPTIndexedBinDataReader(
            data_file_prefix=data_file_prefix,
            seq_length=seq_length,
            num_samples=num_samples,
            batch_size=batch_size,
            shuffle=shuffle,
            random_seed=random_seed,
            device=device,
            placement=placement,
            sbp=sbp,
        )

    def forward(self):
        return self.loader_()


class DataLoaderGraph(flow.nn.Graph):
    def __init__(self, loader):
        super().__init__()
        self.loader_ = loader

    def build(self):
        return self.loader_()


@unittest.skipIf(
    os.getenv("ONEFLOW_TEST_GITHUB_HOSTED"),
    "/dataset not available on GitHub hosted servers",
)
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n2d()
class GPTDataLoaderDistributedTestCase(oneflow.unittest.TestCase):
    def test_case1(test_case):
        rank = flow.env.get_rank()
        # print(
        #     f"GPTDataLoaderDistributedTestCase.test_case1 on rank {rank} {os.getpid()}"
        # )
        eager_gpt_loader = GPTDataLoader(batch_size=4, device=flow.device("cpu", rank))

        global_gpt_loader = GPTDataLoader(
            batch_size=8,
            placement=flow.placement("cpu", ranks=[0, 1]),
            sbp=[flow.sbp.split(0)],
        )
        gpt_loader_graph = DataLoaderGraph(global_gpt_loader)

        iteration = 2
        for i in range(iteration):
            tokens = eager_gpt_loader()
            # print(
            #     f"rank {rank} tokens: {tokens.shape}, {tokens.dtype}, device: {tokens.device}"
            #     f"\n{tokens.numpy()}"
            # )

            g_tokens = gpt_loader_graph()
            # print(
            #     f"rank {rank} graph output tokens: {g_tokens.shape}, {g_tokens.dtype}"
            #     f", placement: {g_tokens.placement}"
            #     f"\n{g_tokens.to_local().numpy()}"
            # )

            # print(f"{'-' * 20} rank {rank} iter {i} complete {'-' * 20}")
            test_case.assertTrue(
                np.allclose(tokens.numpy(), g_tokens.to_local().numpy())
            )


if __name__ == "__main__":
    unittest.main()
