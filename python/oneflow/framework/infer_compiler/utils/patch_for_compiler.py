import math
import torch
import oneflow as flow
import oneflow.nn.functional as F


class FakeCuda:
    @staticmethod
    def current_device():
        return "cuda:0"

    @staticmethod
    def mem_get_info(dev):
        return 1024 * 1024 * 1024, 1024 * 1024 * 1024

    @staticmethod
    def _scaled_dot_product_attention_math(
        query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
    ):
        d_k = query.size(-1)

        if is_causal:
            assert attn_mask is None, "Cannot use both attn_mask and is_causal=True"
            L, S = query.size(-2), key.size(-2)
            attn_mask = flow.ones((L, S), dtype=flow.bool).tril()

        if attn_mask is not None:
            if attn_mask.dtype == flow.bool:
                new_attn_mask = flow.empty(
                    attn_mask.shape, dtype=query.dtype, device=query.device
                )
                mask = flow.logical_not(attn_mask)
                new_attn_mask.masked_fill_(mask, float("-inf"))
                attn_mask = new_attn_mask

        scores = flow.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if attn_mask is not None:
            scores.add_(attn_mask)

        p_attn = F.softmax(scores, dim=-1)

        if dropout_p > 0.0:
            generator = flow.Generator()
            p_attn = flow.nn.functional.dropout(
                p_attn, p=dropout_p, generator=generator
            )

        return flow.matmul(p_attn, value)

    @staticmethod
    def scaled_dot_product_attention(
        query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
    ):
        """Scaled Dot-Product Attention
        Args:
        query (Tensor): Query tensor; shape :math:`(N, ..., L, E)`.
        key (Tensor): Key tensor; shape :math:`(N, ..., S, E)`.
        value (Tensor): Value tensor; shape :math:`(N, ..., S, Ev)`.
        attn_mask (optional Tensor): Attention mask; shape :math:`(N, ..., L, S)`. Two types of masks are supported.
            A boolean mask where a value of True indicates that the element *should* take part in attention.
            A float mask of the same type as query, key, value that is added to the attention score.
        dropout_p (float): Dropout probability; if greater than 0.0, dropout is applied
        is_causal (bool): If true, assumes causal attention masking and errors if both attn_mask and is_causal
            are set.

        Returns:
            output (Tensor): Attention output; shape :math:`(N, ..., L, Ev)`.

        Shape legend:
            - :math:`N: \text{Batch size} ... : \text{Any number of other batch dimensions (optional)}`
            - :math:`S: \text{Source sequence length}`
            - :math:`L: \text{Target sequence length}`
            - :math:`E: \text{Embedding dimension of the query and key}`
            - :math:`Ev: \text{Embedding dimension of the value}`
        """
        if attn_mask is not None or dropout_p > 0.0:
            return FakeCuda._scaled_dot_product_attention_math(
                query, key, value, attn_mask, dropout_p, is_causal
            )

        batch_size, num_heads, target_seq_len, head_dim = query.shape
        out = flow._C.fused_multi_head_attention_inference_v2(
            query=query,
            query_layout="BHMK",
            query_head_size=head_dim,
            key=key,
            key_layout="BHMK",
            value=value,
            value_layout="BHMK",
            output_layout="BM(HK)",
            causal=is_causal,
        )
        # (N, L, H x Ev) -> (N, H, L, Ev)
        value_embed_dim = value.shape[-1]
        out = out.view(batch_size, target_seq_len, num_heads, value_embed_dim).permute(
            0, 2, 1, 3
        )
        return out


flow.cuda.current_device = FakeCuda.current_device
flow.cuda.mem_get_info = FakeCuda.mem_get_info
flow.nn.functional.scaled_dot_product_attention = FakeCuda.scaled_dot_product_attention
F.scaled_dot_product_attention = FakeCuda.scaled_dot_product_attention
