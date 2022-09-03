import json
from typing import Dict, Sequence
import sys

with open(sys.argv[1]) as f:
    x: Dict[str, float] = json.load(f)

def avg(x):
    return sum(map(lambda x: x[0], x)) / sum(map(lambda x: x[1], x))

def get_avg_op_time(ops):
    if not isinstance(ops, Sequence):
        ops = [ops]
    res = []
    for op in ops:
        tmp = [v for k, v in x.items() if k.startswith(f'{op} ')]
        assert len(tmp) > 0
        res.extend(tmp)
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
