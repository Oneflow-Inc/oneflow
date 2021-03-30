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
import numpy as np
import unittest
import oneflow as flow


def _make_gpt_data_loader_func(
    data_file_prefix,
    seq_length,
    num_samples,
    batch_size,
    dtype,
    shuffle=None,
    random_seed=None,
    split_sizes=None,
    split_index=None,
    machine_num=1,
    device_num=1,
    parallel_distribution=None,
):
    assert machine_num > 0
    assert device_num > 0 and device_num <= 4

    parallel_hierachy = None
    if machine_num == 1:
        device_strs = "0:0-{}".format(device_num - 1)
    elif machine_num > 1:
        device_strs = [
            "{}:0-{}".format(machine_id, device_num - 1)
            for machine_id in range(machine_num)
        ]
        parallel_hierachy = (machine_num, device_num)
    else:
        raise ValueError("invalid machine_num", machine_num)

    flow.clear_default_session()
    flow.config.cpu_device_num(4)
    flow.config.enable_legacy_model_io(True)

    func_cfg = flow.FunctionConfig()
    func_cfg.default_logical_view(flow.scope.consistent_view())

    @flow.global_function("predict", function_config=func_cfg)
    def gpt_loader_fn() -> flow.typing.Numpy:
        with flow.scope.placement("cpu", device_strs, parallel_hierachy):
            tokens = flow.data.gpt_data_loader(
                data_file_prefix=data_file_prefix,
                seq_length=seq_length,
                num_samples=num_samples,
                batch_size=batch_size,
                dtype=dtype,
                shuffle=shuffle,
                random_seed=random_seed,
                split_sizes=split_sizes,
                split_index=split_index,
                parallel_distribution=parallel_distribution,
                name="GPTDataLoader",
            )

        return tokens

    check_point = flow.train.CheckPoint()
    check_point.init()
    return gpt_loader_fn


class TestGPTDataLoader(flow.unittest.TestCase):
    DATA_FILE_PREFIX = "/dataset/Megatron-LM/small/gpt_sample_dataset_text_document"
    SEQ_LENGTH = 1024
    RANDOM_SEED = 12345

    @flow.unittest.skip_unless_1n1d()
    def test_1n1d(self):
        of_gpt_data_loader_fn = _make_gpt_data_loader_func(
            data_file_prefix=self.DATA_FILE_PREFIX,
            seq_length=self.SEQ_LENGTH,
            num_samples=8,
            batch_size=8,
            dtype=flow.int64,
            shuffle=True,
            random_seed=self.RANDOM_SEED,
        )
        tokens = of_gpt_data_loader_fn()
        print(tokens.shape)
        print(tokens)

    @flow.unittest.skip_unless_1n4d()
    def test_1n4d(self):
        of_gpt_data_loader_fn = _make_gpt_data_loader_func(
            data_file_prefix=self.DATA_FILE_PREFIX,
            seq_length=self.SEQ_LENGTH,
            num_samples=8,
            batch_size=8,
            dtype=flow.int64,
            shuffle=True,
            random_seed=self.RANDOM_SEED,
            device_num=4,
            parallel_distribution=["S(0)"],
        )
        tokens = of_gpt_data_loader_fn()
        print(tokens.shape)
        print(tokens)


if __name__ == "__main__":
    unittest.main()
