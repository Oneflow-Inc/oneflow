import oneflow as flow
import numpy as np

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)

def _check_unique(test_case, x, y, idx, count, num_unique):
    ref_y, ref_count = np.unique(x, return_counts=True)
    sorted_idx = np.argsort(ref_y)
    ref_y = ref_y[sorted_idx]
    ref_count = ref_count[sorted_idx]
    num_unique = num_unique.item()
    test_case.assertTrue(num_unique, np.size(ref_y))
    y = y[0:num_unique]
    test_case.assertTrue(np.array_equal(y[idx], x))
    sorted_idx = np.argsort(y)
    test_case.assertTrue(np.array_equal(ref_y, y[sorted_idx]))
    count = count[0:num_unique]
    test_case.assertTrue(np.array_equal(count[sorted_idx], ref_count))

def _run_test(test_case, x, dtype, device):
    @flow.function(func_config)
    def UniqueWithCountsJob(x=flow.FixedTensorDef(x.shape, dtype=dtype)):
        with flow.fixed_placement(device, "0:0"):
            return flow.experimental.unique_with_counts(x)
    y, idx, count, num_unique = UniqueWithCountsJob(x).get()
    _check_unique(test_case, x, y.ndarray(), idx.ndarray(), count.ndarray(), num_unique.ndarray())

def test_unique_with_counts_int(test_case):
    x = np.asarray(list(range(32)) * 2).astype(np.int32) 
    np.random.shuffle(x)
    _run_test(test_case, x, flow.int32, 'gpu')
   

def test_unique_with_counts_float(test_case):
    x = np.asarray(list(range(32)) * 2).astype(np.float32) 
    np.random.shuffle(x)
    _run_test(test_case, x, flow.float32, 'gpu')
   
def test_unique_with_counts_random_gpu(test_case):
    x = np.random.randint(0, 32, 1024).astype(np.int32)
    np.random.shuffle(x)
    _run_test(test_case, x, flow.int32, 'gpu')
   
def test_unique_with_counts_random_cpu(test_case):
    x = np.random.randint(0, 32, 1024).astype(np.int32)
    np.random.shuffle(x)
    _run_test(test_case, x, flow.int32, 'cpu')
   

