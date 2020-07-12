# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
common constants
"""

# Built-in supported domains
ONNX_DOMAIN = ""
AI_ONNX_ML_DOMAIN = "ai.onnx.ml"

# Default opset version for onnx domain
PREFERRED_OPSET = 10

NCHW_TO_NHWC = [0, 2, 3, 1]
NHWC_TO_NCHW = [0, 3, 1, 2]
HWCN_TO_NCHW = [3, 2, 0, 1]
NCHW_TO_HWCN = [2, 3, 1, 0]
