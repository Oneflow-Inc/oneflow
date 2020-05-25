# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
common constants
"""

from onnx import helper

TF2ONNX_PACKAGE_NAME = __name__.split('.')[0]

# Built-in supported domains
ONNX_DOMAIN = ""
AI_ONNX_ML_DOMAIN = "ai.onnx.ml"
MICROSOFT_DOMAIN = "com.microsoft"

# Default opset version for onnx domain
PREFERRED_OPSET = 8

# Default opset for custom ops
TENSORFLOW_OPSET = helper.make_opsetid("ai.onnx.converters.tensorflow", 1)

# Target for the generated onnx graph. It possible targets:
# onnx-1.1 = onnx at v1.1 (winml in rs4 is based on this)
# caffe2 = include some workarounds for caffe2 and winml
TARGET_RS4 = "rs4"
TARGET_RS5 = "rs5"
TARGET_RS6 = "rs6"
TARGET_CAFFE2 = "caffe2"
POSSIBLE_TARGETS = [TARGET_RS4, TARGET_RS5, TARGET_RS6, TARGET_CAFFE2]
DEFAULT_TARGET = []

NCHW_TO_NHWC = [0, 2, 3, 1]
NHWC_TO_NCHW = [0, 3, 1, 2]
HWCN_TO_NCHW = [3, 2, 0, 1]
NCHW_TO_HWCN = [2, 3, 1, 0]

# Environment variables
ENV_TF2ONNX_DEBUG_MODE = "TF2ONNX_DEBUG_MODE"
