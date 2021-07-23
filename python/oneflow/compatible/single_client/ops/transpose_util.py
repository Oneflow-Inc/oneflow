from typing import Sequence

def is_perm(perm: Sequence[int]) -> bool:
    return list(range(len(perm))) == sorted(list(perm))

def get_perm_when_transpose_axis_to_last_dim(num_axes: int, axis: int) -> tuple:
    axis = axis if axis >= 0 else axis + num_axes
    assert 0 <= axis < num_axes, 'axis out of range'
    perm = [dim if dim < axis else dim + 1 for dim in range(num_axes - 1)]
    perm.append(axis)
    return tuple(perm)

def get_inversed_perm(perm: Sequence[int]) -> tuple:
    assert is_perm(perm)
    inversed_perm = [-1] * len(perm)
    for i in range(len(perm)):
        inversed_perm[perm[i]] = i
    return tuple(inversed_perm)