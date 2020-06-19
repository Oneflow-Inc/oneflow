import sys

import numpy as np
import oneflow as flow

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)
func_config.default_distribute_strategy(flow.distribute.consistent_strategy())


def test_categorical_ordinal_encoder_gpu(test_case):
    @flow.global_function(func_config)
    def test_job(x=flow.FixedTensorDef((10000,), dtype=flow.int64)):
        return flow.layers.categorical_ordinal_encoder(x, capacity=320000)

    check_point = flow.train.CheckPoint()
    check_point.init()

    tokens = np.random.randint(-sys.maxsize, sys.maxsize, size=[200000]).astype(
        np.int64
    )

    k_set = set()
    v_set = set()
    kv_set = set()
    vk_set = set()
    for i in range(256):
        x = tokens[np.random.randint(0, 200000, (10000,))]
        y = test_job(x).get().ndarray()
        for k, v in zip(x, y):
            k_set.add(k)
            v_set.add(v)
            kv_set.add((k, v))
            vk_set.add((v, k))
    unique_size = len(k_set)
    test_case.assertEqual(len(v_set), unique_size)
    test_case.assertEqual(len(kv_set), unique_size)
    test_case.assertEqual(len(vk_set), unique_size)
