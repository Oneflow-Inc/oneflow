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
    start_from_saved_progress=False,
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
            tokens = flow.data.megatron_gpt_mmap_data_loader(
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
                start_from_saved_progress=start_from_saved_progress,
                name="GPTDataLoader",
            )

            if (
                isinstance(parallel_distribution, list)
                and len(parallel_distribution) > 1
            ):
                tokens = flow.hierarchical_parallel_cast(
                    tokens, parallel_distribution=["B", "B"]
                )

        tokens = flow.hierarchical_parallel_cast(tokens, parallel_distribution=["B"])

        return tokens

    check_point = flow.train.CheckPoint()
    check_point.init()
    return gpt_loader_fn


class TestGPTDataLoader(flow.unittest.TestCase):
    DATA_FILE_PREFIX = "/dataset/Megatron-LM/dummy/gpt_sample_dataset_text_document"
    SEQ_LENGTH = 1024
    RANDOM_SEED = 12345

    @flow.unittest.skip_unless_1n1d()
    def test_simple(self):
        of_gpt_data_loader_fn = _make_gpt_data_loader_func(
            data_file_prefix=self.DATA_FILE_PREFIX,
            seq_length=10,
            num_samples=10,
            batch_size=2,
            dtype=flow.int64,
            shuffle=False,
            start_from_saved_progress=True,
        )
        tokens = of_gpt_data_loader_fn()
        # this comparison tokens is from megatron-lm gpt data loader
        cmp_tokens = np.array(
            [
                [40, 1101, 845, 845, 3772, 13, 428, 318, 257, 1492, 13],
                [13, 612, 318, 257, 18739, 550, 257, 3290, 13, 50256, 464],
            ],
            dtype=np.int64,
        )
        self.assertTrue(np.array_equal(tokens, cmp_tokens))

    def test_1n1d(self):
        of_gpt_data_loader_fn = _make_gpt_data_loader_func(
            data_file_prefix=self.DATA_FILE_PREFIX,
            seq_length=self.SEQ_LENGTH,
            num_samples=648,
            batch_size=8,
            split_sizes=[949, 50, 1],
            split_index=0,
            dtype=flow.int64,
            shuffle=True,
            random_seed=self.RANDOM_SEED,
        )
        tokens_list = []
        for _ in range(5):
            tokens = of_gpt_data_loader_fn()
            tokens_list.append(tokens)

        return np.stack(tokens_list, axis=0)

    @flow.unittest.skip_unless_1n4d()
    def test_1n4d(self):
        of_gpt_data_loader_fn = _make_gpt_data_loader_func(
            data_file_prefix=self.DATA_FILE_PREFIX,
            seq_length=self.SEQ_LENGTH,
            num_samples=648,
            batch_size=8,
            split_sizes=[949, 50, 1],
            split_index=0,
            dtype=flow.int64,
            shuffle=True,
            random_seed=self.RANDOM_SEED,
            device_num=4,
            parallel_distribution=["S(0)"],
        )

        tokens_list = []
        for _ in range(5):
            tokens = of_gpt_data_loader_fn()
            tokens_list.append(tokens)

        result_1n4d = np.stack(tokens_list, axis=0)
        result_1n1d = self.test_1n1d()
        self.assertTrue(np.array_equal(result_1n4d, result_1n1d))
        return result_1n4d

    @flow.unittest.skip_unless_2n4d()
    def test_2n4d(self):
        of_gpt_data_loader_fn = _make_gpt_data_loader_func(
            data_file_prefix=self.DATA_FILE_PREFIX,
            seq_length=self.SEQ_LENGTH,
            num_samples=648,
            batch_size=8,
            split_sizes=[949, 50, 1],
            split_index=0,
            dtype=flow.int64,
            shuffle=True,
            random_seed=self.RANDOM_SEED,
            machine_num=2,
            device_num=4,
            parallel_distribution=["S(0)", "B"],
        )

        tokens_list = []
        for _ in range(5):
            tokens = of_gpt_data_loader_fn()
            tokens_list.append(tokens)

        result_2n4d = np.stack(tokens_list, axis=0)
        result_1n1d = self.test_1n1d()
        self.assertTrue(np.array_equal(result_2n4d, result_1n1d))
        return result_2n4d


if __name__ == "__main__":
    unittest.main()
