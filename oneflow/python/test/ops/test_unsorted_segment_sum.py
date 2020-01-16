import oneflow as flow
import numpy as np
import tensorflow as tf

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)

def _check(test_case, data, segment_ids, num_segments, out):
    ref_out = tf.math.unsorted_segment_sum(data, segment_ids, num_segments).numpy()
    test_case.assertTrue(np.allclose(ref_out, out))

def _run_test(test_case, data, segment_ids, num_segments, data_dtype, segment_id_dtype, device):
    @flow.function(func_config)
    def TestJob(
            data=flow.FixedTensorDef(data.shape, dtype=data_dtype),
            segment_ids=flow.FixedTensorDef(segment_ids.shape, dtype=segment_id_dtype)):
        with flow.fixed_placement(device, "0:0"):
            return flow.math.unsorted_segment_sum(data=data, segment_ids=segment_ids, num_segments=num_segments)
    out = TestJob(data, segment_ids).get()
    _check(test_case, data, segment_ids, num_segments, out.ndarray())

def test_unsorted_segment_sum_gpu(test_case):
    data = np.random.rand(1024, 8).astype(np.float32)
    segment_ids = np.random.randint(0, 32, 1024).astype(np.int32)
    _run_test(test_case, data, segment_ids, 32, flow.float32, flow.int32, 'gpu')


def test_unsorted_segment_sum_cpu(test_case):
    data = np.random.rand(1024, 8).astype(np.float32)
    segment_ids = np.random.randint(0, 32, 1024).astype(np.int32)
    _run_test(test_case, data, segment_ids, 32, flow.float32, flow.int32, 'cpu')

def test_unsorted_segment_sum_gpu_2d(test_case):
    data = np.random.rand(1024, 8).astype(np.float32).reshape([4, 256, 8])
    segment_ids = np.random.randint(0, 32, 1024).astype(np.int32).reshape([4, 256])
    _run_test(test_case, data, segment_ids, 32, flow.float32, flow.int32, 'gpu')


   
def test_unsorted_segment_sum_cpu_2d(test_case):
    data = np.random.rand(1024, 8).astype(np.float32).reshape([4, 256, 8])
    segment_ids = np.random.randint(0, 32, 1024).astype(np.int32).reshape([4, 256])
    _run_test(test_case, data, segment_ids, 32, flow.float32, flow.int32, 'cpu')
 
