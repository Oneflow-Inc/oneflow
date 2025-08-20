import unittest
from collections import OrderedDict

import numpy as np
from itertools import product

import oneflow as flow
import oneflow.unittest
import torch

class TestMultiheadAttention(flow.unittest.TestCase):
    def test_multihead_attention_with_torch(self):
        device_list = ["cpu", "cuda"]
        add_zero_attn_list = [True, False]

        for device, aza in product(device_list, add_zero_attn_list):
            param = {
                "device": device,
                "add_zero_attn": aza,
                "embed_dim": 16,
                "num_heads": 4
            }
            query = np.random.rand(4, 8, param["embed_dim"])
            torch_multihead_attn = torch.nn.MultiheadAttention(
                param["embed_dim"], param["num_heads"], dropout=0.0, add_bias_kv=False, add_zero_attn=param["add_zero_attn"]
            ).to(device)
            torch_query = torch.tensor(query, dtype=torch.float32).to(device)
            y1_torch, y2_torch = torch_multihead_attn(torch_query, torch_query, torch_query)
            flow_multihead_attn = flow.nn.MultiheadAttention(
                param["embed_dim"], param["num_heads"], dropout=0.0, add_bias_kv=False, add_zero_attn=param["add_zero_attn"]
            ).to(device)
            flow_query = flow.tensor(query, dtype=flow.float32).to(device)
            y1_flow, y2_flow = flow_multihead_attn(flow_query, flow_query, flow_query)

            self.assertTrue(
                np.allclose(
                    y1_torch.detach().cpu().numpy(), y1_flow.detach().cpu().numpy(), rtol=1e-4, atol=1e-3
                )
            )
            if y2_torch is None:
                self.assertTrue(y2_flow is None)
            else:
                self.assertTrue(
                    np.allclose(
                        y2_torch.detach().cpu().numpy(), y2_flow.detach().cpu().numpy(), rtol=1e-4, atol=1e-3
                    )
                )

if __name__ == "__main__":
    unittest.main()
