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
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

# Transpose Optimizer

from __future__ import unicode_literals
from collections import defaultdict

import numpy as np
import onnx
from oneflow.python.framework import id_util
from oneflow.python.onnx.constants import NCHW_TO_NHWC, NHWC_TO_NCHW
from .. import util
from .optimizer_base import GraphOptimizerBase


# pylint: disable=logging-not-lazy,unused-argument,missing-docstring,abstract-method
# FIXME:
# pylint: disable=unused-variable


def IsNhwcTranspose(transpose_node):
    perm_attr = transpose_node.attrs.get("perm")
    return transpose_node.op_type == "Transpose" and perm_attr == NCHW_TO_NHWC


def IsNchwTranspose(transpose_node):
    perm_attr = transpose_node.attrs.get("perm")
    return transpose_node.op_type == "Transpose" and perm_attr == NHWC_TO_NCHW


def IsUselessTranspose(transpose_node):
    perm_attr = transpose_node.attrs.get("perm")
    return (
        transpose_node.op_type == "Transpose"
        and perm_attr
        and perm_attr == list(range(len(perm_attr)))
    )


class TransposeOptimizer(GraphOptimizerBase):
    """Transpose Optimizer."""

    def __init__(self):
        super(TransposeOptimizer, self).__init__()

        self._handler_map = {}
        self._force_stop = {}

        self._InitializeHandlers()
        self._g = None
        self._output_names = None

    @property
    def nodes(self):
        return self._g.get_nodes()

    def PreOptimizeAction(self):
        # make Reshape into a const, which then can be fused into Conv's weight for mobilenet_v1_75_192
        self._output_names = [name.split(":")[0] for name in self._g.outputs]
        ops = self.nodes
        constable_reshape_ops = [
            n
            for n in ops
            if (
                n.op_type == "Reshape"
                and n.input_nodes[0].is_const()
                and n.input_nodes[1].is_const()
            )
        ]
        for reshape_op in constable_reshape_ops:
            target_t = reshape_op.input_nodes[0].get_tensor_value(as_list=False)
            target_shape = reshape_op.input_nodes[1].get_tensor_value(as_list=False)
            new_data = np.reshape(target_t, tuple(target_shape))
            const_name = reshape_op.output_tensor_names[0]
            self._g.RemoveNode(reshape_op.name)
            self._g.MakeConst(const_name, new_data)

            # point all children nodes inputs to the new node
            for output_name in reshape_op.output_tensor_names:
                for child in ops:
                    for i, name in enumerate(child.input_tensor_names):
                        if name == output_name:
                            child.input_tensor_names[i] = const_name

            self._g.TopologicalSort(self._g.get_nodes())

    def PoseOptimizeAction(self):
        def _CalculateNewShape(graph, op):
            input_shape = graph.get_shape(op.input_tensor_names[0])
            if input_shape.count(-1) <= 1:
                if IsNchwTranspose(op):
                    new_shape = [
                        input_shape[0],
                        input_shape[3],
                        input_shape[1],
                        input_shape[2],
                    ]
                else:
                    new_shape = [
                        input_shape[0],
                        input_shape[2],
                        input_shape[3],
                        input_shape[1],
                    ]
                return graph.MakeConst(
                    id_util.UniqueStr("new_shape"), np.array(new_shape, dtype=np.int64)
                ).output_tensor_names[0]

            # reshape requires tha output shape can only contain one -1, if not some extra op needed.
            input_shape = graph.MakeNode(
                "Shape", [op.input_tensor_names[0]]
            ).output_tensor_names[0]
            if IsNchwTranspose(op):
                indice = graph.MakeConst(
                    id_util.UniqueStr("indice"), np.array(NHWC_TO_NCHW)
                ).output_tensor_names[0]
            else:
                indice = graph.MakeConst(
                    id_util.UniqueStr("indice"), np.array(NCHW_TO_NHWC)
                ).output_tensor_names[0]

            return graph.MakeNode("Gather", [input_shape, indice]).output_tensor_names[
                0
            ]

        nodes = self.nodes
        # if channel==1 or height==width==1, replace transpose with reshape
        # replacing trans with reshape is because transpose will copy data even if this transpose doesn't nothing
        for op in nodes:
            if op.op_type == "Transpose":
                input_shape = self._g.get_shape(op.input_tensor_names[0])
                if not input_shape:
                    continue

                if (
                    IsNchwTranspose(op)
                    and (input_shape[3] == 1 or (input_shape[1:3] == [1, 1]))
                ) or (
                    IsNhwcTranspose(op)
                    and (input_shape[1] == 1 or (input_shape[2:4] == [1, 1]))
                ):
                    new_shape = _CalculateNewShape(self._g, op)
                    # replace transpose with reshape
                    self._g.RemoveNode(op.name)
                    self._g.MakeNode(
                        "Reshape",
                        [op.input_tensor_names[0], new_shape],
                        name=op.name,
                        outputs=op.output_tensor_names,
                    )
                    self._g.TopologicalSort(self._g.get_nodes())

    def MergeDuplicatedTransposes(self):
        # strategy used in previous procedure is to move transpose nodes down if possible,
        # and it means that when a node has n outputs then n transpose will be generated,
        # so we should merge them back to one if they can't be eliminated in previous procedure.
        graph = self._g
        input_transposes_map = defaultdict(list)
        for node in graph.get_nodes():
            if node.op_type == "Transpose" and node.attrs.get("perm"):
                key = (node.input_tensor_names[0], str(node.attrs["perm"]))
                input_transposes_map[key].append(node)

        for transposes in input_transposes_map.values():
            # merge transpose nodes into one: make nodes use the output of the first transpose node
            transpose_out = transposes[0].output_tensor_names[0]
            for node in transposes[1:]:
                old_transpose_out = node.output_tensor_names[0]
                graph.ReplaceAllInputs(
                    graph.get_nodes(), old_transpose_out, transpose_out
                )

        # dangling transpose nodes can be deleted
        graph.DeleteUnusedNodes(graph.outputs)

    def _Optimize(self, graph):
        return self._ApplyOptimization(graph, self._OptimizeAtCurrentGraphLevel)

    def _OptimizeAtCurrentGraphLevel(self, graph):
        self._g = graph
        self.PreOptimizeAction()
        no_action = False
        iteration_cnt = 0
        while not no_action:
            no_action = True
            nodes = self.nodes
            self._force_stop = {}
            for n in nodes:
                if IsNhwcTranspose(n):
                    if self._HandleNhwcTranspose(n):
                        no_action = False
                        self.graph_been_opt = True
                        iteration_cnt += 1
                        # need break, because handler may change nodes set, making the n stale object
                        # referencing already deleted elements
                        break

                if IsUselessTranspose(n):
                    no_action = False
                    iteration_cnt += 1
                    self._RemoveUselessTranpose(n)
                    break
            # for debugging purpose
            if "stop" in self._force_stop and self._force_stop["stop"] == 1:
                break

        self.logger.debug("finish after " + str(iteration_cnt) + " iteration(s)")

        self.MergeDuplicatedTransposes()
        self.PoseOptimizeAction()
        return self._g

    def _InitializeHandlers(self):
        self._handler_map = {
            "Add": self._AddHandler,
            "Cast": self._SimpleThroughHandler,
            "Clip": self._SimpleThroughHandler,
            "Concat": self._ConcatHandler,
            "Identity": self._IdentityHandler,
            "LeakyRelu": self._SimpleThroughHandler,
            "Max": self._MaxminHandler,
            "Min": self._MaxminHandler,
            "Mul": self._MulHandler,
            "Pad": self._PadHandler,
            "ReduceMean": self._ReducemeanHandler,
            "Relu": self._SimpleThroughHandler,
            "Shape": self._ShapeHandler,
            "Slice": self._SliceHandler,
            "Split": self._SplitHandler,
            "Squeeze": self._SqueezeHandler,
            "Sub": self._SubHandler,
            "Tanh": self._SimpleThroughHandler,
            "Transpose": self._TransposeHandler,
        }

    def _HandleNodeHavingBranches(self, node):
        # create transpose pairs if some input are not.
        if not self._CreateTransposePairsBeforeNode(node):
            return False
        # make sure node's all input transpose all have only 1 consumer node,
        # otherwise, it would impact their other output nodes
        if (
            self._NodesHasSingleConsumerNode(node.input_nodes)
            and len(node.output_tensor_names) == 1
        ):
            self._CreateTransposePairsAfterNode(node)
            input_transposes = set(node.input_nodes)
            for n in input_transposes:
                n_input = n.input_tensor_names[0]
                util.MakeSure(
                    len(n.output_tensor_names) == 1, "only expect single output"
                )
                self._g.ReplaceAllInputs(
                    self._g.get_nodes(), n.output_tensor_names[0], n_input
                )
                self._g.RemoveNode(n.name)

            util.MakeSure(
                len(node.output_tensor_names) == 1, "only expect single output"
            )
            # currently we assume node only has 1 output, for cases where it is more than 1 for example Split
            # we need consider the fact that Split's multiple output will not always has data in NCHW/NHWC,
            # it might be a different shape.
            output_transposes = self._g.FindOutputConsumers(node.output_tensor_names[0])
            for n in output_transposes:
                n_input = n.input_tensor_names[0]
                util.MakeSure(
                    len(n.output_tensor_names) == 1, "only expect single output"
                )
                self._g.ReplaceAllInputs(
                    self._g.get_nodes(), n.output_tensor_names[0], n_input
                )
                self._g.RemoveNode(n.name)

            shape = self._g.get_shape(node.output_tensor_names[0])
            if shape:
                # only nhwc transpose can reach here
                new_shape = [shape[i] for i in NHWC_TO_NCHW]
                self._g.set_shape(node.output_tensor_names[0], new_shape)
            return True

        self.logger.debug("input transpose does not have single consumer, skipping...")
        return False

    # get the input index of transpose op in node's inputs.
    def _GetInputIndexForTrans(self, node, trans):
        input_index = 0
        for i in node.input_tensor_names:
            if i == trans.output_tensor_names[0]:
                break
            input_index += 1
        return input_index

    # the assumption is: both node and trans have only 1 output
    def _SwitchTransposeAndNode(self, node, trans):
        if not self._NodesHasSingleConsumerNode([trans]):
            return False

        input_index = self._GetInputIndexForTrans(node, trans)

        ops = self._g.get_nodes()
        self._g.ReplaceAllInputs(
            ops, node.output_tensor_names[0], trans.output_tensor_names[0]
        )
        node.input_tensor_names[input_index] = trans.input_tensor_names[0]
        trans.input_tensor_names[0] = node.output_tensor_names[0]

        # need to transpose node shape in backward direction as well after switch
        # otherwise, reshape added in PoseOptimizeAction may not work correctly
        shape = self._g.get_shape(node.output_tensor_names[0])
        if shape:
            # only nhwc transpose can reach here
            new_shape = [shape[i] for i in NHWC_TO_NCHW]
            self._g.set_shape(node.output_tensor_names[0], new_shape)
        return True

    # if return value is True, then it means Transpose is handled as designed
    # otherwise, it means that we skip handling since it is not in our support set
    def _HandleNhwcTranspose(self, trans):
        if trans.output_tensor_names[0] in self._g.outputs:
            self.logger.debug(
                "%s connects to graph outputs, skip", trans.output_tensor_names[0]
            )
            return False
        out_nodes = self._g.FindOutputConsumers(trans.output_tensor_names[0])
        if len(out_nodes) == 1:
            p = out_nodes[0]
            if p.name in self._output_names:
                self.logger.debug(
                    "cannot move transpose down since it met output node %s", p.name
                )
                return False

            if p.op_type in self._handler_map:
                op_handler = self._handler_map[p.op_type]
                return op_handler(trans, p)
            return False
        # move transpose into branches to let Transposes can be "handled" in each branch
        for n in out_nodes:
            branch_trans = self._g.MakeNode(
                "Transpose", [trans.input_tensor_names[0]], attr=trans.attrs
            )
            self._g.ReplaceAllInputs(
                n, trans.output_tensor_names[0], branch_trans.output_tensor_names[0]
            )

        self._g.RemoveNode(trans.name)
        return False

    def _RemoveUselessTranpose(self, trans):
        self._g.ReplaceAllInputs(
            self._g.get_nodes(),
            trans.output_tensor_names[0],
            trans.input_tensor_names[0],
        )
        self._g.RemoveNode(trans.name)

    def _NodesHasSingleConsumerNode(self, nodes):
        for n in nodes:
            for output in n.output_tensor_names:
                cnt = len(set(self._g.FindOutputConsumers(output)))
                if cnt != 1:
                    return False
        return True

    def _GetNonNchwTransposeOutputNodes(self, node):
        # we just support node having 1 output, we need consider cases where node has more than 1 outputs
        assert len(node.output_tensor_names) == 1
        non_nchw_tranpose_nodes = []
        consumers = self._g.FindOutputConsumers(node.output_tensor_names[0])
        for o in consumers:
            if not IsNchwTranspose(o) and o not in non_nchw_tranpose_nodes:
                non_nchw_tranpose_nodes.append(o)
        return non_nchw_tranpose_nodes

    def _CreateTransposePairsAfterNode(self, node):
        assert len(node.output_tensor_names) == 1  # just support node who has 1 output
        non_nchw_trans_consumers = self._GetNonNchwTransposeOutputNodes(node)
        # add Transpose(0, 3, 1, 2) and Transpose(0, 2, 3, 1) before each non_nchw_trans_consumers
        for consumer in non_nchw_trans_consumers:
            nchw_node = self._g.MakeNode(
                "Transpose", [node.output_tensor_names[0]], attr={"perm": NHWC_TO_NCHW}
            )
            nhwc_node = self._g.MakeNode(
                "Transpose",
                [nchw_node.output_tensor_names[0]],
                attr={"perm": NCHW_TO_NHWC},
            )
            self._g.ReplaceAllInputs(
                consumer, node.output_tensor_names[0], nhwc_node.output_tensor_names[0]
            )

    def _CreateTransposePairsBeforeNode(self, node):
        def shape_after_expand(ori_shape):
            # according to broadcasting rule to expand shape to 4D while not tile the tensor here
            # still count on the broadcasting op to tile the tensor
            if ori_shape.count(-1) >= 2:
                self.logger.warning(
                    "%s shape can contain one -1 at most, otherwise reshape op can't work",
                    node.name,
                )
                return None
            ori_rank = len(ori_shape)
            new_shape = [1] * (4 - ori_rank) + ori_shape
            return new_shape

        non_nhwc_trans_inputs = []
        for input_id, n in zip(node.input_tensor_names, node.input_nodes):
            if not IsNhwcTranspose(n):
                # check in case node has two inputs coming from a same node output.
                if [input_id, n] not in non_nhwc_trans_inputs:
                    non_nhwc_trans_inputs.append([input_id, n])

        # add Transpose(0, 3, 1, 2) and Transpose(0, 2, 3, 1) before each non_nhwc_trans_consumers
        shape_unknow = [
            input_id
            for input_id, _ in non_nhwc_trans_inputs
            if self._g.get_shape(input_id) is None
        ]
        if shape_unknow:
            if self._g.opset <= 9:
                msg = (
                    "%s 's shape is unknown, ConstantOfShape will be used which exists in version 9 or higher"
                    "while graph's opset version is %s" % (shape_unknow, self._g.opset)
                )
                self.logger.warning(msg)
                return False

        for input_id, n in non_nhwc_trans_inputs:
            shape = self._g.get_shape(input_id)
            # if rank of n is not 4, then we need to insert a reshape op before inserting a transpose
            # for example shape of n is [x, y], then output shape of reshape will be [1, 1, x, y]
            if shape is None:
                const_4 = self._g.MakeConst(
                    id_util.UniqueStr("const_4"), np.array([4], np.int64)
                ).output_tensor_names[0]
                tensor_1 = onnx.helper.make_tensor(
                    "value", onnx.TensorProto.INT64, [1], [1]
                )
                shape_node = self._g.MakeNode("Shape", [input_id]).output_tensor_names[
                    0
                ]
                rank_node = self._g.MakeNode("Shape", [shape_node]).output_tensor_names[
                    0
                ]
                expand_rank = self._g.MakeNode(
                    "Sub", [const_4, rank_node]
                ).output_tensor_names[0]
                array_fill_1 = self._g.MakeNode(
                    "ConstantOfShape", [expand_rank], attr={"value": tensor_1}
                ).output_tensor_names[0]
                new_shape = self._g.MakeNode(
                    "Concat", [array_fill_1, shape_node], attr={"axis": 0}
                ).output_tensor_names[0]
                reshape = self._g.MakeNode(
                    "Reshape", [input_id, new_shape]
                ).output_tensor_names[0]
                input_of_new_trans = reshape
            elif len(shape) == 4:
                input_of_new_trans = input_id
            else:
                shape_4d = shape_after_expand(shape)
                if shape_4d is None:
                    return False
                const = self._g.MakeConst(
                    id_util.UniqueStr("reshape_shape"), np.array(shape_4d, np.int64)
                ).output_tensor_names[0]
                reshape = self._g.MakeNode(
                    "Reshape", [input_id, const]
                ).output_tensor_names[0]
                input_of_new_trans = reshape

            nchw_node = self._g.MakeNode(
                "Transpose", [input_of_new_trans], attr={"perm": NHWC_TO_NCHW}
            )
            nhwc_node = self._g.MakeNode(
                "Transpose",
                [nchw_node.output_tensor_names[0]],
                attr={"perm": NCHW_TO_NHWC},
            )
            self._g.ReplaceAllInputs(node, input_id, nhwc_node.output_tensor_names[0])
        return True

    def _AddHandler(self, trans, node):
        if node.input_nodes[1].is_const():
            t_p = trans.input_nodes[0]
            if (
                t_p.op_type in ("Conv", "ConvTranspose")
                and len(t_p.input_tensor_names) == 2
            ):
                # if Conv or ConvTranspose's bias input is not set, then we set, otherwise, we don't set

                if not self._NodesHasSingleConsumerNode([t_p]):
                    self.logger.debug(
                        "Conv does not have single consumer, can not merge Conv and Add"
                    )
                    return self._HandleNodeHavingBranches(node)

                if not self._NodesHasSingleConsumerNode([trans]):
                    self.logger.debug(
                        "input transpose does not have single consumer, skipping..."
                    )
                    return False

                target_node = node.input_nodes[1]
                numpy_val = target_node.get_tensor_value(as_list=False)
                # Optional 1D bias to be added to the convolution, has size of M
                if len(numpy_val.shape) - numpy_val.shape.count(1) > 1:
                    self.logger.debug("Bias is not 1D, can not merge Conv and Add")
                    return self._HandleNodeHavingBranches(node)

                bias_size = max(numpy_val.shape)
                size_m = t_p.input_nodes[1].output_shapes[0][0]
                if bias_size != size_m:
                    self.logger.debug("Bias size is not M, can not merge Conv and Add")
                    return self._HandleNodeHavingBranches(node)

                target_val = numpy_val.reshape(bias_size)
                target_node.set_tensor_value(target_val)

                conv_inputs = [
                    t_p.input_tensor_names[0],
                    t_p.input_tensor_names[1],
                    node.input_tensor_names[1],
                ]
                conv_node = self._g.MakeNode(t_p.op_type, conv_inputs, attr=t_p.attrs)
                ops = self._g.get_nodes()
                trans.input_tensor_names[0] = id_util.UniqueStr(conv_node.name)
                self._g.ReplaceAllInputs(
                    ops, node.output_tensor_names[0], trans.output_tensor_names[0]
                )
                self._g.RemoveNode(t_p.name)
                self._g.RemoveNode(node.name)
                return True
        return self._HandleNodeHavingBranches(node)

    def _TransposeHandler(self, trans, node):
        if IsNchwTranspose(node):
            for g in {self._g, node.graph}:
                ops = g.get_nodes()
                g.ReplaceAllInputs(
                    ops, node.output_tensor_names[0], trans.input_tensor_names[0]
                )

            shape = node.graph.get_shape(node.output_tensor_names[0])
            dtype = node.graph.get_dtype(node.output_tensor_names[0])
            if node.output_tensor_names[0] in node.graph.outputs:
                node.graph.MakeNode(
                    "Identity",
                    [trans.input_tensor_names[0]],
                    outputs=node.output_tensor_names,
                    shapes=[shape],
                    dtypes=[dtype],
                )
            self._g.RemoveNode(trans.name)
            node.graph.RemoveNode(node.name)
            return True
        return False

    def _MaxminHandler(self, trans, node):
        return self._HandleNodeHavingBranches(node)

    def _MulHandler(self, trans, node):
        multiplier_input_id = None
        multiplier_input_node = None
        for i, input_node in zip(node.input_tensor_names, node.input_nodes):
            if i != trans.output_tensor_names[0]:
                multiplier_input_id = i
                multiplier_input_node = input_node

        # node's inputs may come from one same node. if so the multiplier_input_node may be none
        if multiplier_input_node is None or not multiplier_input_node.is_const():
            return False
        multiplier = multiplier_input_node.get_tensor_value(as_list=False)

        if multiplier_input_id == node.input_tensor_names[1]:
            t_p = trans.input_nodes[0]
            # make sure conv don't have bias set
            if (
                t_p.op_type == "Conv"
                and t_p.input_nodes[1].is_const()
                and len(t_p.input_tensor_names) == 2
            ):
                conv = t_p
                numpy_val = conv.input_nodes[1].get_tensor_value(as_list=False)
                transposed_val = np.transpose(numpy_val, (2, 3, 1, 0))
                mul_val = multiplier
                result = np.multiply(transposed_val, mul_val)
                conv.input_nodes[1].set_tensor_value(np.transpose(result, (3, 2, 0, 1)))

                ops = self._g.get_nodes()
                self._g.ReplaceAllInputs(
                    ops, node.output_tensor_names[0], trans.output_tensor_names[0]
                )
                self._g.RemoveNode(node.name)
                return True

        # if the shape is () or (1), we just move transpose after the mul
        if not multiplier.shape or (
            len(multiplier.shape) == 1 and multiplier.shape[0] == 1
        ):
            return self._SwitchTransposeAndNode(node, trans)

        return False

    def _IdentityHandler(self, trans, node):
        if node.output_tensor_names[0] in node.graph.outputs:
            return False
        for g in {self._g, node.graph}:
            ops = g.get_nodes()
            g.ReplaceAllInputs(
                ops, node.output_tensor_names[0], trans.output_tensor_names[0]
            )
        node.graph.RemoveNode(node.name)
        return True

    def _ConcatHandler(self, trans, node):
        if self._HandleNodeHavingBranches(node):
            perm = trans.attrs["perm"]
            axis = node.attrs.get("axis", 0)
            new_axis = perm[axis]
            node.attrs["axis"] = new_axis
            return True
        return False

    def _SplitHandler(self, trans, node):
        # TODO(daquexian): need handle cases where Split node has more than 1 outputs.
        if self._HandleNodeHavingBranches(node):
            node.attrs["axis"] = 1
            return True
        return False

    def _SqueezeHandler(self, trans, node):
        def _CalculateNewAttr(ori_perm, ori_squeeze_axes):
            new_squeeze_axes = sorted([ori_perm[i] for i in ori_squeeze_axes])
            # calculate output shape after trans and squeeze
            input_shape = "abcd"
            shape_after_trans = [input_shape[i] for i in ori_perm]
            output_shape = [
                shape_after_trans[i] for i in range(4) if i not in ori_squeeze_axes
            ]
            # calculate new_perm
            # after switch, the output shape should be same, using this condtion we can figure the new perm
            shape_after_squeeze = [
                input_shape[i] for i in range(4) if i not in new_squeeze_axes
            ]
            new_perm = [shape_after_squeeze.index(i) for i in output_shape]

            return new_perm, new_squeeze_axes

        if not self._NodesHasSingleConsumerNode([trans]):
            return False

        if "axes" in node.attrs:
            # switch tran and squeeze
            # 1 switch
            ops = self._g.get_nodes()
            self._g.ReplaceAllInputs(
                ops, node.output_tensor_names[0], trans.output_tensor_names[0]
            )
            node.input_tensor_names[0] = trans.input_tensor_names[0]
            trans.input_tensor_names[0] = node.output_tensor_names[0]
            # 2 correct attr of nodes
            squeeze_axes = sorted(node.attrs["axes"])
            trans_perm = trans.attrs["perm"]
            new_perm, new_squeeze_axes = _CalculateNewAttr(
                ori_perm=trans_perm, ori_squeeze_axes=squeeze_axes
            )
            trans.attrs["perm"] = new_perm
            node.attrs["axes"] = new_squeeze_axes
            # 3 set shape
            squeeze_shape = self._g.get_shape(node.output_tensor_names[0])
            self._g.set_shape(trans.output_tensor_names[0], squeeze_shape)
            input_shape = self._g.get_shape(node.input_tensor_names[0])
            if input_shape is not None:
                new_squeeze_output_shape = [
                    input_shape[i] for i in range(4) if i not in new_squeeze_axes
                ]
            else:
                new_squeeze_output_shape = [-1] * 4
                self.logger.warning(
                    "%s's shape is unknown, which may interfere further optimization",
                    node.input_tensor_names[0],
                )
            self._g.set_shape(node.output_tensor_names[0], new_squeeze_output_shape)
            return True
        return False

    def _SubHandler(self, trans, node):
        return self._HandleNodeHavingBranches(node)

    def _PadHandler(self, trans, node):
        # [N-start, H-start, W-start, C-start, N-end, H-end,  W-end, C-end]
        if self._g.opset < 11:
            pads = node.attrs["pads"]  # [x1_begin, x2_begin...x1_end, x2_end,...]
            # NHWC->NCHW
            new_pads = [
                pads[0],
                pads[3],
                pads[1],
                pads[2],
                pads[4],
                pads[7],
                pads[5],
                pads[6],
            ]
            node.attrs["pads"] = new_pads
            return self._SwitchTransposeAndNode(node, trans)
        if node.input_nodes[1].is_const():
            pads = node.input_nodes[1].get_tensor_value()
            # NHWC->NCHW
            new_pads = np.array(
                [
                    pads[0],
                    pads[3],
                    pads[1],
                    pads[2],
                    pads[4],
                    pads[7],
                    pads[5],
                    pads[6],
                ],
                dtype=np.int64,
            )
            node.input_nodes[1].set_tensor_value(new_pads)
            return self._SwitchTransposeAndNode(node, trans)
        return False

    def _ReducemeanHandler(self, trans, node):
        axes = node.attrs["axes"]
        keepdims = node.attrs.get("keepdims", 1)
        # make sure keepdims is 1, then we can do the swap, otherwise, please don't, because
        # once keepdims is not set, original dims are lost, so transpose back won't work well.
        # by default, if keepdims is not specified, it is 1
        if axes == [1, 2] and keepdims == 1:
            node.attrs["axes"] = [2, 3]
            return self._SwitchTransposeAndNode(node, trans)
        return False

    def _SliceHandler(self, trans, node):
        axes = None
        if self._g.opset < 10:
            axes = node.attrs["axes"]
            if axes == [0, 1, 2, 3]:
                node.attrs["axes"] = NCHW_TO_NHWC
                return self._SwitchTransposeAndNode(node, trans)
        else:  # in opset 10, axes is input instead of an attribute.
            if len(node.input_nodes) >= 4 and node.input_nodes[3].is_const():
                axes = node.input_nodes[3].get_tensor_value(as_list=True)
                if axes == [0, 1, 2, 3]:
                    # axes node might be shared
                    new_axes = np.array(NCHW_TO_NHWC, dtype=np.int64)
                    if self._NodesHasSingleConsumerNode([node]):
                        node.input_nodes[3].set_tensor_value(new_axes)
                    else:
                        new_axes_const = self._g.MakeConst(
                            id_util.UniqueStr(node.input_nodes[3].name), new_axes
                        )
                        self._g.ReplaceAllInputs(
                            node,
                            node.input_tensor_names[3],
                            new_axes_const.output_tensor_names[0],
                        )
                    return self._SwitchTransposeAndNode(node, trans)
        return False

    def _SimpleThroughHandler(self, trans, node):
        return self._SwitchTransposeAndNode(node, trans)

    def _ShapeHandler(self, trans, node):
        # input > trans > shape  can be changed into  input > shape > gather
        if not self._NodesHasSingleConsumerNode([trans]):
            return False

        output_shape = self._g.get_shape(node.output_tensor_names[0])
        output_dtype = self._g.get_dtype(node.output_tensor_names[0])
        self._g.RemoveNode(trans.name)
        self._g.RemoveNode(node.name)
        shape_node = self._g.MakeNode("Shape", [trans.input_tensor_names[0]])
        const_node = self._g.MakeConst(
            id_util.UniqueStr("Const"), np.array(trans.attrs["perm"])
        )
        gather_node = self._g.MakeNode(
            "Gather",
            [shape_node.output_tensor_names[0], const_node.output_tensor_names[0]],
            outputs=node.output_tensor_names,
        )
        self._g.set_shape(gather_node.output_tensor_names[0], output_shape)
        self._g.set_dtype(gather_node.output_tensor_names[0], output_dtype)
        return True
