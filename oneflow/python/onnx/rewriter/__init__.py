# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.
"""oneflow.python.onnx.rewriter module."""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from oneflow.python.onnx.rewriter.cond_rewriter import rewrite_cond
from oneflow.python.onnx.rewriter.conv2d_with_pad_rewriter import rewrite_conv2d_with_pad
from oneflow.python.onnx.rewriter.dropout_rewriter import rewrite_dropout
from oneflow.python.onnx.rewriter.eye_rewriter import rewrite_eye
from oneflow.python.onnx.rewriter.flatten_rewriter import rewrite_flatten
from oneflow.python.onnx.rewriter.gemm_rewriter import rewrite_gemm
from oneflow.python.onnx.rewriter.leakyrelu_rewriter import rewrite_leakyrelu
from oneflow.python.onnx.rewriter.random_normal_rewriter import rewrite_random_normal
from oneflow.python.onnx.rewriter.random_uniform import rewrite_random_uniform, rewrite_random_uniform_fold_const
from oneflow.python.onnx.rewriter.rnn import rewrite_single_direction_lstm, rewrite_bi_direction_lstm, \
    rewrite_single_direction_gru, rewrite_bi_direction_gru, \
    rewrite_custom_rnn_cell, rewrite_generic_loop
from oneflow.python.onnx.rewriter.thresholded_relu_rewriter import rewrite_thresholded_relu
from oneflow.python.onnx.rewriter.transpose_rewriter import rewrite_transpose

__all__ = [
    "rewrite_cond",
    "rewrite_conv2d_with_pad",
    "rewrite_dropout",
    "rewrite_eye",
    "rewrite_flatten",
    "rewrite_gemm",
    "rewrite_leakyrelu",
    "rewrite_random_normal",
    "rewrite_random_uniform",
    "rewrite_random_uniform_fold_const",
    "rewrite_thresholded_relu",
    "rewrite_transpose",

    "rewrite_single_direction_lstm",
    "rewrite_bi_direction_lstm",
    "rewrite_single_direction_gru",
    "rewrite_bi_direction_gru",
    "rewrite_custom_rnn_cell",
    "rewrite_generic_loop",
]
