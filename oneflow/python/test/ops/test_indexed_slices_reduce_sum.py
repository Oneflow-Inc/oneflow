import oneflow as flow
import numpy as np

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)

def _check(test_case, x_indices, x_values, y_indices, y_values, num_unique):
    ref_indices = np.unique(x_indices)
    np.sort(ref_indices)
    num_unique = num_unique.item()
    test_case.assertTrue(num_unique == ref_indices.shape[0])
    key_to_idx = dict(zip(ref_indices, range(num_unique)))
    ref_values = np.zeros((num_unique, y_values.shape[-1]), y_values.dtype)
    for i in range(x_indices.shape[0]):
        ref_values[key_to_idx[x_indices[i].item()]] += x_values[i]
    y_indices = y_indices[0:num_unique]
    y_values = y_values[0:num_unique]
    sorted_idx = np.argsort(y_indices)
    y_indices = y_indices[sorted_idx]
    y_values = y_values[sorted_idx]
    test_case.assertTrue(np.array_equal(ref_indices, y_indices))
    test_case.assertTrue(np.allclose(ref_values, y_values))

def _run_test(test_case, indices, values, indices_dtype, values_dtype, device):
    @flow.function(func_config)
    def TestJob(
            indices=flow.FixedTensorDef(indices.shape, dtype=indices_dtype), 
            values=flow.FixedTensorDef(values.shape, dtype=values_dtype)):
        with flow.fixed_placement(device, "0:0"):
            return flow.experimental.indexed_slices_reduce_sum(indices, values)
    out_indices, out_values, num_unique = TestJob(indices, values).get()
    _check(test_case, indices, values, out_indices.ndarray(), out_values.ndarray(), num_unique.ndarray())

def test_indexed_slices_reduce_sum_gpu(test_case):
    indices = np.random.randint(0, 32, 1024).astype(np.int32)
    values = np.random.rand(1024, 8).astype(np.float32)
    _run_test(test_case, indices, values, flow.int32, flow.float32, 'gpu')


def test_indexed_slices_reduce_sum_cpu(test_case):
    indices = np.random.randint(0, 32, 1024).astype(np.int32)
    values = np.random.rand(1024, 8).astype(np.float32)
    _run_test(test_case, indices, values, flow.int32, flow.float32, 'cpu')
   
