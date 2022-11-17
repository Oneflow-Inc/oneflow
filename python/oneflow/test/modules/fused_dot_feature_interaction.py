import numpy as np
import oneflow as flow

def fused_dot_feature_interaction(x,
                                  y,
                                  self_interaction=False,
                                  output_padding=0,
                                  output_concat=None,
                                  dtype=flow.float32
                                  ):
    # (bs, es) = x.shape
    (bs, dims, es) = y.shape
    
    if self_interaction:
        offset = 1
    else:
        offset = 0
    li = flow.tensor([i for i in range(dims + 1) for j in range(i + offset)])
    lj = flow.tensor([j for i in range(dims + 1) for j in range(i + offset)])
    T = flow.cat(
        [
            flow.reshape(x, (bs, 1, es)),
            y,
        ],
        dim=1,
    )
    Z = flow.matmul(T, T, transpose_b=True)
    # gather_nd not support half, so cast to float32
    Z = flow.cast(Z, flow.float32)
    Zflat = Z[:, li, lj]
    Zflat = flow.cast(Zflat, dtype)
    if output_concat is not None:
        R = flow.cat([output_concat, Zflat], dim=1)
    else:
        R = Zflat
    if output_padding != 0:
        padding_tensor = flow.tensor(
            np.zeros((bs, output_padding)).astype(np.float32),
            device="cuda",
            requires_grad=False,
        )
        R = flow.cat([R, padding_tensor], dim=1)
    return R
