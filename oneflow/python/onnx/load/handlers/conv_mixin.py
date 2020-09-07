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
import oneflow as flow

from oneflow.python.ops import nn_ops
from .broadcast_mixin import BroadcastMixin
from .pad_mixin import PadMixin


class ConvMixin(BroadcastMixin):
    @classmethod
    def conv(cls, node, input_dict, transpose=False):
        """ Convolution method for both conv and transposed conv
    For transposed conv,
      Attr pads is not used for input, but declares how much output is padded.
      Here, output means output from transposed conv which already pad output_padding if set.
      So the pseudo explanation for output should be:
        output = conv_transpose_output + output_padding - pads
      And conv_transpose_output shape should be:
        conv_transpose_output_shape[i] = strides[i] * (input_shape[i] - 1) + kernel_shape[i]
    """
        x = input_dict[node.input_tensor_names[0]]
        x_shape = list(x.shape)
        x_rank = len(x_shape)
        spatial_size = x_rank - 2

        in_weights = input_dict[node.input_tensor_names[1]]
        in_weights_shape = list(in_weights.shape)
        weights_rank = len(in_weights_shape)
        if transpose:
            # Translate weights from (C x M x KH x KW) to (KH x KW X M X C)
            perm = list(range(2, weights_rank)) + [1, 0]
        else:
            # Translate weights from (M x C x KH x KW) to (KH x KW X C X M)
            perm = list(range(2, weights_rank)) + [1, 0]

        if "kernel_shape" in node.attrs.keys():
            kernel_shape = node.attrs["kernel_shape"]
            assert in_weights_shape[2:] == kernel_shape, (
                "kernel_shape "
                "attr of convolution does not match the actual weight "
                "passed to this operation, attr {}, actual {}"
            ).format(kernel_shape, in_weights_shape)
        else:
            kernel_shape = in_weights_shape[2:]

        weights = in_weights
        dilations = node.attrs.get("dilations", [1] * spatial_size)
        strides = node.attrs.get("strides", [1] * spatial_size)

        pads = node.attrs.get("pads", [0, 0] * spatial_size)

        # Check auto_pad nonexistent or NOTSET first
        if "auto_pad" not in node.attrs or node.attrs["auto_pad"] == "NOTSET":
            if not transpose:
                if pads != [0, 0] * spatial_size:
                    x = PadMixin.get_padding_as_op(x, pads)
                pad_mode = "VALID"
            else:
                pad_mode = "NOTSET"
        # Then we use auto_pad to setup pad_mode
        elif node.attrs["auto_pad"] == "SAME_UPPER":
            pad_mode = "SAME"
        elif node.attrs["auto_pad"] == "VALID":
            pad_mode = "VALID"
        elif node.attrs["auto_pad"] == "SAME_LOWER":
            pad_mode = "SAME_LOWER"
        else:
            raise ValueError(
                "Invalid auto_pad attribute: {}".format(node.attrs["auto_pad"])
            )

        group = node.attrs.get("group", 1)

        conv = nn_ops.conv2d(
            x,
            weights,
            padding=pad_mode,
            strides=strides,
            dilations=dilations,
            data_format="NCHW",
            groups=group,
        )

        if len(node.input_tensor_names) == 2:
            output = conv
        else:
            bias = input_dict[node.input_tensor_names[2]]
            output = nn_ops.bias_add(conv, bias)

        return [output]
