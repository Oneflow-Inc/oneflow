import json
from typing import Dict, Sequence, List, Tuple
import sys
import functools

with open(sys.argv[1]) as f:
    x: Dict[str, Tuple[float, int]] = json.load(f)

def avg(x):
    return sum(map(lambda x: x[0], x)) / sum(map(lambda x: x[1], x))

def get_avg_op_time(ops):
    if not isinstance(ops, Sequence):
        ops = [ops]
    res = []
    for op in ops:
        for k, v in x.items():
            if k.startswith(f'{op}|'):
                output_shape_str = k.split('|')[2].split(', ')[0]
                output_shape = [int(x) for x in output_shape_str.strip('()').split(',')]
                output_elem_cnt = functools.reduce(lambda x, y: x * y, output_shape)
                # print(f'{op}: {output_shape} {output_elem_cnt} {v}')
                res.append((v[0] / (output_elem_cnt * 4 / 1024 / 1024), v[1]))
                # res.append((v[0], v[1]))
    return avg(res)

transformer = int(sys.argv[2]) != 0
if transformer:
    print(f"matmul and backward: {get_avg_op_time(['batch_matmul', 'broadcast_matmul'])}")
    print(f"ln and backward: {get_avg_op_time(['layer_norm'])}")
    print(f"gelu and backward: {get_avg_op_time(['gelu'])}")
else:
    print(f"conv and backward: {get_avg_op_time(['conv2d'])}")
    print(f"bn and backward: {get_avg_op_time(['normalization'])}")
    print(f"relu and backward: {get_avg_op_time(['relu'])}")
