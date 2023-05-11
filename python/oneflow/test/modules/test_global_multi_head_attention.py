import unittest
import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *
import numpy as np
import torch

def _test_multi_head_attention_class_with_random_data(
    test_case, placement, input_sbp, m, name, embed_dim, query, graph
):
    query = (
        m.from_numpy(query).to(m.float32).to_global(placement=placement, sbp=input_sbp)
        if name == "flow"
        else m.from_numpy(query).to(m.float32)
    )
    key = query
    value = query
    num_heads = 4

    dropout_p = 0  # must be 0
    bias = True

    multihead_attention = m.nn.MultiheadAttention(
        embed_dim=embed_dim, num_heads=num_heads, dropout=dropout_p, bias=bias
    )

    if name == "flow":
        multihead_attention.to_global(placement=placement, sbp=input_sbp)

    if graph:

        class GraphModel(flow.nn.Graph):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def build(self, input=None, key=None, value=None):
                return self.module(input, key, value)

        func = GraphModel(multihead_attention)
        return func(query, key, value)

    y1, y2 = multihead_attention(query, key, value)
    return y1, y2

class TestMultiHeadAttentionClass(flow.unittest.TestCase):
    @globaltest
    def test_multi_head_attention_class_with_random_data(test_case):
        embed_dim = 16
        for placement in all_placement():
            for input_sbp in all_sbp(placement, max_dim=2):
                for graph in [False, True]:
                    query = np.random.rand(4, 8, embed_dim)
                    y1_flow, y2_flow = _test_multi_head_attention_class_with_random_data(
                        test_case,
                        placement,
                        input_sbp,
                        flow,
                        "flow",
                        embed_dim,
                        query.copy(),
                        graph,
                    )
                    y1_torch, y2_torch = _test_multi_head_attention_class_with_random_data(
                        test_case,
                        placement,
                        input_sbp,
                        torch,
                        "torch",
                        embed_dim,
                        query.copy(),
                        False,
                    )

                    test_case.assertTrue(
                        np.allclose(
                            y1_torch.detach().cpu().numpy(),
                            y1_flow.detach().cpu().numpy(),
                            rtol=1e-4,
                            atol=1e-3,
                        )
                    )
                    if y2_torch is None:
                        test_case.assertTrue(y2_flow is None)
                    else:
                        test_case.assertTrue(
                            np.allclose(
                                y2_torch.detach().cpu().numpy(),
                                y2_flow.detach().cpu().numpy(),
                                rtol=1e-4,
                                atol=1e-3,
                            )
                        )

if __name__ == "__main__":
    unittest.main()
