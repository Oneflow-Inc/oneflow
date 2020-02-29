import oneflow as flow
import numpy as np

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)
func_config.default_distribute_strategy(flow.distribute.consistent_strategy())


def test_categorical_hash_table_gpu(test_case):
    @flow.function(func_config)
    def CategoricalHashTable(x=flow.FixedTensorDef((100000,), dtype=flow.int64)):
        return flow.experimental.layers.categorical_hash_table(x, capacity=3200000)

    check_point = flow.train.CheckPoint()
    check_point.init()

    k_set = set()
    v_set = set()
    kv_set = set()
    vk_set = set()
    for i in range(256):
        x = np.random.randint(0, 2000000, (100000,)).astype(np.int64)
        y = CategoricalHashTable(x).get().ndarray()
        for k, v in zip(x, y):
            k_set.add(k)
            v_set.add(v)
            kv_set.add((k, v))
            vk_set.add((v, k))
    unique_size = len(k_set)
    test_case.assertEqual(len(v_set), unique_size)
    test_case.assertEqual(len(kv_set), unique_size)
    test_case.assertEqual(len(vk_set), unique_size)
