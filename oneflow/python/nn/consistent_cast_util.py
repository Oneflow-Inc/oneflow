"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import absolute_import
from collections import OrderedDict, namedtuple
from typing import Union, Optional, Tuple, List, Callable
import oneflow as flow
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.session_context as session_ctx
from oneflow._oneflow_internal.one import (
    CastToConsistentOpExpr,
    CastFromConsistentOpExpr,
    OpExpr,
)
import oneflow._oneflow_internal


def cast_input_to_consistent(input_tensors, parallel_distribution, placement_signature):
    input_tensors = list(input_tensors)
    sess = session_ctx.GetDefaultSession()
    cast_to_consistent_ops = _make_cast_consistent_ops(
        parallel_distribution,
        placement_signature,
        CastToConsistentOpExpr,
        id_util.UniqueStr("cast_to_consistent_op"),
    )
    assert len(input_tensors) == len(cast_to_consistent_ops)
    with sess.consistent_scope():
        for i in range(0, len(input_tensors)):
            if not input_tensors[i].is_consistent:
                input_tensors[i] = cast_to_consistent_ops[i](input_tensors[i])[0]
    return tuple(input_tensors)


def cast_output_from_consistent(
    output_tensors, parallel_distribution, placement_signature
):
    # assume type of output_tensors is TensorTuple or Tensor
    if isinstance(output_tensors, oneflow._oneflow_internal.TensorTuple):
        output_tensors = list(output_tensors)
    elif isinstance(output_tensors, oneflow._oneflow_internal.Tensor):
        output_tensors = [output_tensors]
    else:
        raise NotImplementedError
    sess = session_ctx.GetDefaultSession()
    cast_from_consistent_ops = _make_cast_consistent_ops(
        parallel_distribution,
        placement_signature,
        CastFromConsistentOpExpr,
        id_util.UniqueStr("cast_from_consistent_op"),
    )
    assert len(output_tensors) == len(cast_from_consistent_ops)
    with sess.consistent_scope():
        for i in range(0, len(output_tensors)):
            if output_tensors[i].is_consistent:
                output_tensors[i] = cast_from_consistent_ops[i](output_tensors[i])[0]
    return output_tensors[0] if len(output_tensors) == 1 else tuple(output_tensors)


def _make_cast_consistent_ops(
    sbp_signature: List[Union[Tuple[str], str]],
    placement: List[flow.placement],
    op_expr: OpExpr,
    name: Optional[str] = None,
) -> None:
    assert len(sbp_signature) == len(placement)
    cast_consistent_op = OrderedDict()
    for i in range(len(sbp_signature)):
        parallel_distribution = sbp_signature[i]
        if isinstance(parallel_distribution, str):
            parallel_distribution = [parallel_distribution]
        else:
            assert isinstance(parallel_distribution, tuple)
            parallel_distribution = list(parallel_distribution)
        assert len(parallel_distribution) == len(placement[i].hierarchy)
        cast_consistent_op_expr = op_expr(
            name if name is not None else id_util.UniqueStr("cast_consistent"),
            parallel_distribution,
            placement[i],
        )
        cast_consistent_op[i] = cast_consistent_op_expr
    return cast_consistent_op
