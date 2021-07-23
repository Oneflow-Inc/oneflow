import unittest
import oneflow as flow
import numpy as np

@flow.unittest.skip_unless_1n2d()
class TestAllReduce(flow.unittest.TestCase):

    def test_all_reduce(test_case):
        arr_rank1 = np.array([1, 2])
        arr_rank2 = np.array([3, 4])
        if flow.distributed.get_rank() == 0:
            x = flow.Tensor([1, 2])
        elif flow.distributed.get_rank() == 1:
            x = flow.Tensor([3, 4])
        else:
            raise ValueError
        x = x.to(f'cuda:{flow.distributed.get_local_rank()}')
        nccl_allreduce_op = flow.builtin_op('eager_nccl_all_reduce').Input('in').Output('out').Attr('parallel_conf', f'device_tag: "gpu", device_name: "0:0-1"').Build()
        y = nccl_allreduce_op(x)[0]
        test_case.assertTrue(np.allclose(y.numpy(), arr_rank1 + arr_rank2))
if __name__ == '__main__':
    unittest.main()