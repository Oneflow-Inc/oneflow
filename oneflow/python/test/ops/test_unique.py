import oneflow as flow
import numpy as np

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)

def test_unique_with_counts(test_case):
    @flow.function(func_config)
    def UniqueWithCountsJob(x=flow.FixedTensorDef((64,), dtype=flow.int32)):
        return flow.experimental.unique_with_counts(x)
   
    x = np.asarray(list(range(32)) * 2).astype(np.int32) 
    np.random.shuffle(x)
    y, idx, count, num_unique = UniqueWithCountsJob(x).get()
    num_unique = num_unique.ndarray().item()
    test_case.assertTrue(num_unique, 32)
    y = y.ndarray()[0:num_unique]
    idx = idx.ndarray();
    test_case.assertTrue(np.array_equal(y[idx], x))
    np.sort(y)
    test_case.assertTrue(np.array_equal(np.asarray(list(range(32))).astype(np.int32), y))
    count = count.ndarray()[0:num_unique]
    test_case.assertTrue(np.all(count == 2))
