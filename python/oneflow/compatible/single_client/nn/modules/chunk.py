from typing import Optional
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.framework.tensor import Tensor
from oneflow.compatible.single_client.python.framework.tensor import register_tensor_op
from oneflow.compatible.single_client.python.nn.module import Module
from oneflow.compatible.single_client.python.ops.array_ops import check_slice_tup_list


class Chunk(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input, chunks, dim):
        if dim is not None:
            assert input.shape[dim] > 0, "chunk expects at least a 1-dimensional tensor"
            assert chunks > 0, "chunk expects `chunks` to be greater than 0"
            channel = input.dim()
            dim_size = input.shape[dim]
            chunk_size = (
                dim_size / chunks if dim_size % chunks == 0 else int(dim_size / chunks)
            )
            last_chunk_size = (
                dim_size / chunks
                if dim_size % chunks == 0
                else dim_size - chunk_size * (chunks - 1)
            )
            chunk_dim_dict = {}
            tup_ndim = []
            splits = []
            for chunk in range(0, chunks):
                if dim_size % chunks == 0:
                    start = chunk * chunk_size
                    stop = (chunk + 1) * chunk_size
                else:
                    start = (
                        chunk * chunk_size
                        if chunk < chunks - 1
                        else chunk_size * (chunks - 1)
                    )
                    stop = (chunk + 1) * chunk_size if chunk < chunks - 1 else dim_size
                step = 1
                chunk_dim_dict.setdefault(dim, []).append(
                    [int(start), int(stop), int(step)]
                )
            for (k, v) in chunk_dim_dict.items():
                for v_chunk in v:
                    tup_list = []
                    for i in range(0, channel):
                        if i != dim:
                            tup_list.append([None, None, None])
                        else:
                            tup_list.append(v_chunk)
                    (start_tup, stop_tup, step_tup) = check_slice_tup_list(
                        tup_list, input.shape
                    )
                    splits.append(
                        flow.F.slice(
                            input, start=start_tup, stop=stop_tup, step=step_tup
                        )
                    )
            return splits


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
