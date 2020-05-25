# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
oneflow.python.onnx.rewriter.rnn_utils - rnn support
"""

from __future__ import unicode_literals
from collections import defaultdict
from enum import Enum

import logging
import numpy as np
from oneflow.python.onnx import utils
from oneflow.python.onnx.graph_builder import GraphBuilder
from oneflow.python.onnx.graph_matcher import OpTypePattern, GraphMatcher # pylint: disable=unused-import


# pylint: disable=invalid-name,unused-argument,missing-docstring



logger = logging.getLogger(__name__)


class REWRITER_RESULT(Enum):
    SKIP = 1
    OK = 2
    FAIL = 3


# TensorFlow LSTMCell/BasicLSTMCell computation graph matching
xc_pattern = OpTypePattern('Split', inputs=[
    OpTypePattern("Const"), # axis for split
    OpTypePattern("BiasAdd", name="bias_add", inputs=[
        OpTypePattern("MatMul", inputs=[
            OpTypePattern("ConcatV2|Concat", name="xh"),
            OpTypePattern("Enter", inputs=[
                OpTypePattern("*", name="cell_kernel"),
            ]),
        ]),
        OpTypePattern("Enter", inputs=[
            OpTypePattern("*", name="cell_bias"),
        ]),
    ]),
])


lstmcell_pattern = \
    OpTypePattern('Mul', name='ht', inputs=[
        OpTypePattern("Sigmoid", name="ot", inputs=[xc_pattern]),
        OpTypePattern('Tanh', inputs=[
            OpTypePattern("Add", name="ct", inputs=[
                OpTypePattern("Mul", name="ct_identity_consumer", inputs=[
                    OpTypePattern("Sigmoid", name="ft", inputs=[
                        OpTypePattern("Add", inputs=[
                            xc_pattern,
                            OpTypePattern("*", name="ft_bias"),
                        ]),
                    ]),
                    OpTypePattern("*"),
                ]),
                OpTypePattern("Mul", inputs=[
                    OpTypePattern("Sigmoid", name="it", inputs=[xc_pattern]),
                    OpTypePattern("Tanh", name="gt", inputs=[xc_pattern]),
                ]),
            ]),
        ]),
    ])

# input sequence: top to down, left to right
# split into update gate and reset gate
gru_split_pattern = \
    OpTypePattern("Split", inputs=[
        OpTypePattern("Const"),  # split dim, a constant
        OpTypePattern("Sigmoid", inputs=[
            OpTypePattern("BiasAdd", inputs=[
                OpTypePattern("Enter", inputs=[
                    OpTypePattern("*", name="gate_bias")
                ]),
                OpTypePattern("MatMul", name="update_reset_gate", inputs=[
                    OpTypePattern("Enter", inputs=[
                        OpTypePattern("*", name="gate_kernel")
                    ]),
                    OpTypePattern("ConcatV2|Concat", name="cell_inputs")
                ])
            ])
        ])
    ])


grucell_pattern = \
    OpTypePattern("Add", name="cell_output", inputs=[
        OpTypePattern("Mul", inputs=[
            gru_split_pattern,
            OpTypePattern("Identity")
        ]),
        OpTypePattern("Mul", inputs=[
            OpTypePattern("Sub", inputs=[
                OpTypePattern("Const"),  # 1-u
                gru_split_pattern
            ]),
            OpTypePattern("*", name="optional_activation", inputs=[
                OpTypePattern("BiasAdd", inputs=[
                    OpTypePattern("Enter", inputs=[
                        OpTypePattern("*", name="hidden_bias")
                    ]),
                    OpTypePattern("MatMul", inputs=[
                        OpTypePattern("Enter", inputs=[
                            OpTypePattern("*", name="hidden_kernel")
                        ]),
                        OpTypePattern("ConcatV2|Concat")
                    ])
                ])
            ])
        ])
    ])


cudnn_compatible_grucell_pattern = \
    OpTypePattern("Add", name="cell_output", inputs=[
        OpTypePattern("Mul", inputs=[
            OpTypePattern("Sub", inputs=[
                OpTypePattern("Const"),  # 1-u
                gru_split_pattern
            ]),
            OpTypePattern("*", name="optional_activation", inputs=[
                OpTypePattern("Add", inputs=[
                    OpTypePattern("Mul", inputs=[
                        gru_split_pattern,
                        OpTypePattern("BiasAdd", inputs=[
                            OpTypePattern("Enter", inputs=[
                                OpTypePattern("*", name="hidden_state_bias")
                            ]),
                            OpTypePattern("MatMul", inputs=[
                                OpTypePattern("Enter", inputs=[
                                    OpTypePattern("*", name="hidden_state_kernel"),
                                ]),
                                OpTypePattern("Identity")
                            ])
                        ])
                    ]),
                    OpTypePattern("BiasAdd", inputs=[
                        OpTypePattern("Enter", inputs=[
                            OpTypePattern("*", name="hidden_input_bias")
                        ]),
                        OpTypePattern("MatMul", inputs=[
                            OpTypePattern("Enter", inputs=[
                                OpTypePattern("*", name="hidden_input_kernel"),
                            ]),
                            OpTypePattern("*")
                        ])
                    ])
                ])
            ])
        ]),
        OpTypePattern("Mul", inputs=[
            gru_split_pattern,
            OpTypePattern("Identity")
        ])
    ])


grublockcell_pattern = OpTypePattern("GRUBlockCell", name="gru_block_cell", inputs=[
    OpTypePattern("*"),
    OpTypePattern("*"),
    OpTypePattern("Enter", inputs=[
        OpTypePattern("*", name="gate_kernel")
    ]),
    OpTypePattern("Enter", inputs=[
        OpTypePattern("*", name="hidden_kernel")
    ]),
    OpTypePattern("Enter", inputs=[
        OpTypePattern("*", name="gate_bias")
    ]),
    OpTypePattern("Enter", inputs=[
        OpTypePattern("*", name="hidden_bias")
    ])
])


lstmblockcell_pattern = \
    OpTypePattern("LSTMBlockCell", name="lstm_block_cell", inputs=[
        OpTypePattern("*"),
        OpTypePattern("*"),
        OpTypePattern("*"),
        OpTypePattern("Enter", inputs=[
            OpTypePattern("*", name="cell_kernel")
        ]),
        OpTypePattern("*", name="Pi"),
        OpTypePattern("*", name="Pf"),
        OpTypePattern("*", name="Po"),
        OpTypePattern("Enter", inputs=[
            OpTypePattern("*", name="cell_bias")
        ])
    ])



seq_len_pattern = OpTypePattern("Select", inputs=[
    OpTypePattern("GreaterEqual", inputs=[
        OpTypePattern("*"),
        OpTypePattern("Enter", inputs=[
            OpTypePattern("*", name="seq_len_node")
        ])
    ]),
    OpTypePattern("*"),
    OpTypePattern("*")
])


class RNNUnitType(Enum):
    LSTMCell = 0  # TF LSTMCell and BasicLSTMCell share the same pattern
    LSTMBlockCell = 1
    GRUCell = 2
    GRUBlockCell = 3
    CudnnCompatibleGRUCell = 4


rnn_cell_patterns = {
    RNNUnitType.LSTMCell: lstmcell_pattern,
    RNNUnitType.LSTMBlockCell: lstmblockcell_pattern,
    RNNUnitType.GRUCell: grucell_pattern,
    RNNUnitType.GRUBlockCell: grublockcell_pattern,
    RNNUnitType.CudnnCompatibleGRUCell: cudnn_compatible_grucell_pattern
}


def get_pattern(cell_type_name):
    return rnn_cell_patterns[cell_type_name]


def get_rnn_scope_name(while_scope_name):
    parts = while_scope_name.split('/')
    rnn_scope = '/'.join(parts[0:-2]) + "/"
    return rnn_scope


def parse_rnn_loop(graph, loop_properties, rnn_scope, while_context_scope):
    """check if the while loop is generated by dynamic_rnn or bidirectional_rnn

    Args:
        loop_properties: LoopProperties
        rnn_scope: rnn scope name
        while_context_scope: while loop scope name

    check a while loop is generated by dynamic_rnn or bidirectional_rnn by

    1. some patterns in _time_step in dynamic_rnn: tensor array read, tensor array write
    2. some patterns in control_flow_ops.while_loop in dynamic_rnn:
         cond: time < loop_bound
         loop_vars: (time, output_ta, state)
         time has name called "time"
         iteration_cnt is added by control flow.

    be noted:
    1. iteration counter does not exist in tf1.4 or earlier versions
    2. if dynamic_rnn's first input is not consumed, output ta does not exist.
    """
    time_name = rnn_scope + "time"
    ta_array_name_prefix = rnn_scope + "dynamic_rnn/output_"
    iteration_counter_name = while_context_scope + "iteration_counter"

    found_time = False
    is_rnn_out_ta = None
    time_var = None
    iteration_var = None
    for val in loop_properties.all_variables.values():
        enter_input_node = graph.get_node_by_output(val.enter_input_id)
        if val.is_tensor_array:
            ta_name = enter_input_node.get_attr("tensor_array_name").s.decode("utf-8")
            if not ta_name.startswith(ta_array_name_prefix):
                is_rnn_out_ta = False
        elif enter_input_node.name == time_name:
            found_time = True
            time_var = val
        elif enter_input_node.name == iteration_counter_name:
            iteration_var = val

    if not found_time or is_rnn_out_ta is False:
        logger.debug("this should not be a dynamic_rnn loop, found_time: %s, is_rnn_out_ta: %s",
                     found_time, is_rnn_out_ta)
        return None

    if not loop_properties.tensor_array_inputs:
        logger.debug("this should not be a dynamic_rnn loop, no ta input is found")
        return None

    return time_var, iteration_var


def get_weights_from_const_node(g, node):
    temp = node
    val = None
    # this would help ignore Identity in non-const_folded graph.
    while temp.type == 'Identity':
        temp = temp.inputs[0]

    if temp and temp.type == 'Const':
        val = temp.get_tensor_value(as_list=False)
        dtype = utils.map_onnx_to_numpy_type(g.get_dtype(temp.output[0]))
        val = val.astype(dtype)
        logger.debug("found weights %s", temp.name)
    else:
        logger.debug("weight node seems not to be Const, skip, node name is %s", temp.name)
        return None

    return val


######################################################
####      Utilities for bidirectional rnn      #######
######################################################
class ONNX_RNN_TYPE(Enum):
    GRU = 0
    LSTM = 1


onnx_rnn_type_mapping = {
    ONNX_RNN_TYPE.GRU: "GRU",
    ONNX_RNN_TYPE.LSTM: "LSTM"
}

onnx_rnn_attr_mapping = {
    ONNX_RNN_TYPE.LSTM: [
        "clip",
        "hidden_size",
        "input_forget"
    ],
    ONNX_RNN_TYPE.GRU: {
        "clip",
        "hidden_size",
        "linear_before_reset"
    }
}
onnx_rnn_seq_len_index_mapping = {
    ONNX_RNN_TYPE.LSTM: 4,
    ONNX_RNN_TYPE.GRU: 4
}


def find_bidirectional_rnns(g, ops, rnn_type):
    """
    Find possible bidirectional rnns, return: list of tuple,
    Format of tuple is (fw onnx rnn node, bw onnx rnn node).
    """
    fw_rnns = defaultdict(list)
    bw_rnns = defaultdict(list)
    for n in g.get_nodes():
        if n.type != onnx_rnn_type_mapping[rnn_type]:
            continue

        input_id = n.input[0]
        temp = n.inputs[0]
        is_bw = False
        if temp.type == "Transpose":
            input_id = temp.input[0]
            temp = temp.inputs[0]

        if utils.is_tf_reverse_op(temp):
            input_id = temp.input[0]
            is_bw = True

        if is_bw:
            # if output 0 is consumed and there is no reverse after the 1st output.
            # it's not backward rnn.
            if g.find_output_consumers(n.output[0]) and not get_reverse_nodes_after_y_output(g, n):
                logger.warning("rnn %s following Reverse op isn't the part of bi-rnn.", n.name)
                continue

            logger.debug("find bw rnn %s", input_id)
            bw_rnns[input_id].append(n)
        else:
            logger.debug("find fw rnn %s", input_id)
            fw_rnns[input_id].append(n)

    # fw_rnn and bw_rnn must share the same input
    birnn_input = list(set(fw_rnns.keys()).intersection(bw_rnns.keys()))
    bi_rnns = []
    matched_rnn = []
    for inp in birnn_input:
        fw_rnn = fw_rnns[inp]
        bw_rnn = bw_rnns[inp]
        # it's possible several bi-rnns share the same input
        for fw_n in fw_rnn:
            for bw_n in bw_rnn:
                if belong_to_birnn(g, fw_n, bw_n, rnn_type) and \
                        not fw_n in matched_rnn and not bw_n in matched_rnn:
                    logger.debug("found birnn comprising %s and %s", fw_n.name, bw_n.name)
                    bi_rnns.append((fw_n, bw_n))
                    matched_rnn.extend([fw_n, bw_n])
    return bi_rnns


def belong_to_birnn(g, fw_rnn, bw_rnn, rnn_type):
    """
    Check whether fw_rnn and bw_rnn are part of the same birnn.
    If fw_rnn and bw_rnn have the same attributes except those related to activation
    and share the same seq_len, they are able to be merged into a bi-rnn.
    """
    logger.debug("check whether %s and %s are part of birnn", fw_rnn.name, bw_rnn.name)
    for name in onnx_rnn_attr_mapping[rnn_type]:
        fw_attr_value = fw_rnn.get_attr_value(name)
        bw_attr_value = bw_rnn.get_attr_value(name)
        if fw_attr_value != bw_attr_value:
            logger.debug(
                "fw_rnn and bw_rnn mismatch at attr %s: %s, %s",
                name, fw_attr_value, bw_attr_value
            )
            return False

    seq_len_index = onnx_rnn_seq_len_index_mapping[rnn_type]
    fw_seq_len = fw_rnn.input[seq_len_index]
    bw_seq_len = bw_rnn.input[seq_len_index]
    if not utils.have_same_inference_value(g, fw_seq_len, bw_seq_len):
        logger.debug(
            "fw_rnn and bw_rnn have different seq_len input: %s, %s",
            fw_seq_len, bw_seq_len
        )
        return False

    return True


def get_reverse_nodes_after_y_output(g, rnn_bw):
    bw_consumers = g.find_output_consumers(rnn_bw.output[0])

    # todo: figure out a better way to remove reverse op
    squeeze_nodes = [c for c in bw_consumers if c.type == "Squeeze"]
    s_cnt = len(squeeze_nodes)
    if s_cnt == 1:
        s = squeeze_nodes[0]
        trans_nodes = g.find_output_consumers(s.output[0])
        if len(trans_nodes) == 1:
            if trans_nodes[0].type == "Transpose":
                reverse_nodes = g.find_output_consumers(trans_nodes[0].output[0])
            elif utils.is_tf_reverse_op(trans_nodes[0]):
                reverse_nodes = trans_nodes
            else:
                logger.debug("not found reverse op, unexpected")
                return []

            are_all_reverse = all([utils.is_tf_reverse_op(r_op) for r_op in reverse_nodes])
            if are_all_reverse:
                return reverse_nodes

            logger.debug("bw y output is used followed by reverse node")
            return []

        logger.debug("unexpected number of transpose after RNN 1st output:%s", s_cnt)
        return []

    logger.debug("unexpected number of squeeze following RNN 1st output:%s", s_cnt)
    return []


def get_np_val_for_const(g, node, input_index):
    return node.inputs[input_index].get_tensor_value(as_list=False)


def check_const(g, input_id):
    node = g.get_node_by_output(input_id)
    if node and node.is_const():
        return (True, node.get_tensor_value(as_list=False))
    return (None, None)


def process_single_init_node(g, fw_init_input_id, bw_init_input_id, to_append):
    fw_init_is_const, init_fw_val = check_const(g, fw_init_input_id)
    bw_init_is_const, init_bw_val = check_const(g, bw_init_input_id)
    if fw_init_is_const and bw_init_is_const:
        initial_val = np.concatenate((init_fw_val, init_bw_val), axis=0)
        init_name = utils.make_name("initial")
        init_node = g.make_const(init_name, initial_val, skip_conversion=True)
    else:
        init_node = g.make_node("Concat", [fw_init_input_id, bw_init_input_id], attr={"axis": 0})

    to_append.append(init_node)
    return init_node


def slice_birnn_for_original_rnn_consumers(g, rnn_fw, rnn_bw, bi_rnn, rnn_output_index, all_nodes, to_remove):
    fw_consumers = g.find_output_consumers(rnn_fw.output[rnn_output_index])
    bw_consumers = g.find_output_consumers(rnn_bw.output[rnn_output_index])
    if not fw_consumers and not bw_consumers:
        return

    if rnn_output_index == 0:
        axis = 1
        # remove reverse op for rnn_bw
        reverse_nodes = get_reverse_nodes_after_y_output(g, rnn_bw)

        for r_op in reverse_nodes:
            logger.debug("remove reverse op %s", r_op.name)
            g.replace_all_inputs(all_nodes, r_op.output[0], r_op.input[0])
            to_remove.append(r_op.name)
    elif rnn_output_index in [1, 2]:
        axis = 0
    else:
        raise ValueError("rnn only should has 3 outputs.")

    if fw_consumers:
        attr = {"axes": [axis], "starts": [0], "ends": [1]}
        inputs_map = {"data": bi_rnn.output[rnn_output_index], **attr}
        slice_node_fw = GraphBuilder(g).make_slice(inputs_map)
        all_nodes.append(g.get_node_by_output(slice_node_fw))
        g.replace_all_inputs(fw_consumers, rnn_fw.output[rnn_output_index], slice_node_fw)

    if bw_consumers:
        attr = {"axes": [axis], "starts": [1], "ends": [2]}
        inputs_map = {"data": bi_rnn.output[rnn_output_index], **attr}
        slice_node_bw = GraphBuilder(g).make_slice(inputs_map)
        all_nodes.append(g.get_node_by_output(slice_node_bw))
        g.replace_all_inputs(bw_consumers, rnn_bw.output[rnn_output_index], slice_node_bw)


def remove_reverse_in_bw_input(g, bw_rnn_input_x, rnn_type):
    old_x_consumers = g.find_output_consumers(bw_rnn_input_x)
    # the transpose/reverse here must be followed by RNN if it is still useful.
    # this is guaranteed by dynamic_rnn logic.
    old_x_has_rnn_as_consumer = [n for n in old_x_consumers if n.type == onnx_rnn_type_mapping[rnn_type]]
    if not old_x_has_rnn_as_consumer:
        logger.debug("plan to remove useless reverse op in bw")
        reverse_node = g.get_node_by_output(bw_rnn_input_x)

        if reverse_node.type == "Transpose":
            reverse_node = reverse_node.inputs[0]

        g.replace_all_inputs(g.get_nodes(), reverse_node.output[0], reverse_node.input[0])
        g.remove_node(reverse_node.name)
    else:
        raise ValueError("Reverse is still used by RNN as input, cannot remove")
