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
from __future__ import division

import tensorflow as tf
import numpy as np

from oneflow.python.ops import nn_ops, pad
from oneflow.python.onnx.load.common import pooling_helper
from oneflow.python.onnx.load.common.tf_helper import tf_shape, flow_shape
from oneflow.python.onnx.load.common.tf_helper import tf_product


class DilatedPooling(object):
    """
        This class implements two main methods:
            dilated_pool:
                calculates a max or average pool over the input

            dilated_maxpool_with_argmax:
                calculates a maxpool over the input and returns the
                indices/argmax of the selected values

        In addition to the standard features of pooling operations in
        Tensorflow, these methods support dilations, ceil mode, SAME_LOWER and
        explicit padding.

        Dilations are partly supported in Tensorflow in `tf.nn.pool` and
        `tf.nn.dilation2d`. The code will try to use the Tensoflow build-in
        functions as much as poosible.

        In cases, not supported by Tensorflow there is a custom algorith of
        dilated pooling `_remove_dilations`.

        The idea behind `_remove_dilations` is to transform the input N-D data
        into a supported input for the standard tf.nn.pool operation.
        This is achieved by calculating N-D indicies for the values which will
        be selected from the input when applying the dilations and
        then extracting the values using tf.gather_nd. Next step is to execute
        `tf.nn.pool` on this new input data with **strides=kernel_shape** and
        no dilations. The resulting pool will be the result we are looking for.

        In case of `deilated_maxpool_with_argmax` an additional step is needed
        to recalculated the resulting indices back into the original
        data indices. It is done with `_calc_orig_argmax`

        Here is a simple example of how the algorithm works:

        kernel_shape = [3]
        strides = [2]
        dilations = [3]

        Input 1D data:

            x-----x-----x-----x-----x-----x-----x-----x-----x-----x-----x
            |  *  |     | **  |  *  |     | **  |  *  |     | **  |     |
            | 10  |  9  | 30  |  7  |  6  | 15  | 16  | 17  | 18  | 19  |
            x-----x-----x-----x-----x-----x-----x-----x-----x-----x-----x
              (0)   (1)   (2)   (3)   (4)   (5)   (6)   (7)   (8)   (9)

        where * represents the values selected during the first sliding window
        step and ** during the second sliding window step

        the resulting indices will be:

            [0, 3, 6, 2, 5, 8]
             |     |  |     |
              First    Second
              step     step

        after tf.gather_nd operation we get a new input data with
        removed dilations:

            [10, 7, 16, 30, 15, 18]

        and apllying tf.nn.maxpool (or avgpool) with strides = kernel_shape = 3
        will result into:

            [16, 30]

        which is the result of the dilated maxpooling.

        Here is pseudo code of the algorithm with comments:

        FUNCTION _remove_dilations:
            /* Calculate N-D index of the values to be selected by the
               dilations and strides */

            /* Do a loop over the input spatial dimensions starting from the
               last (most internal) going up to the first dimension

               On every step of the loop calculate the input indices and
               "combine" them with the already calculated indices from the
               previous dimensions using cartesian product.
            */
            LOOP with **dimension** from **dimensions_count** to **0**:

                // Initialize empty gather_nd index
                gather_ind = []

                // Calculate the output size for the current dimension
                dim_filter_size = (dim_kernel_size - 1) * dim_dilations
                dim_output_size = (((dim_input_size - dim_filter_size) //
                                   dim_strides) + 1) * dim_kernel_size)

                /* For every output index, calculate the corresponding index
                   into the input data */
                dim_input_indices = range(0, dim_output_size)
                dim_input_indices = calculate_input_indicies(dim_input_indices)

                /* combine the calculated indices with the previous dimensions
                */
                gather_ind = cartesian_product(dim_input_indices, gather_ind)
            END LOOP

            /* For example for 2D input the resulting gather_ind will
               look like this:

               [[y1, x1], [y2, x2], ..., [yn, xm]]

               where:
               n is the height
               m is the width and
               [xi, yi] are the 2D indices in the input data
            */

            new_data = tf.gather_nd(input, gather_ind)

            reshape new_data to the correct output shape

            RETURN new_data


        Before executing _remove_dilations the code will apply paddings to the
        input data if needed. Padding is done using tf.pad with -inf values.
        Check `_remove_dilations` code for more details explanation of the
        implementation

        In case of dilated_maxpool_with_argmax the returned indices from
        tf.nn.max_pool_with_argmax will point into our "no dilations" data.
        That is why they need to be mapped back to the original input data.
        It is done with `_calc_orig_argmax` function which will apply the same
        calculations, that are used in _remove_dilations when calculating the
        input data indices from output indices (check `_calc_orig_argmax` for
        detailed inline comments explaining the calculations)

    """

    def __init__(
        self,
        input,
        kernel_shape,
        strides,
        dilations,
        padding="VALID",
        ceil_mode=False,
        count_include_pad=False,
        pooling_type="MAX",
    ):
        self.input = input

        self.kernel_shape = kernel_shape
        self.strides = strides
        self.dilations = dilations
        self.padding = padding
        self.is_explicit_padding = type(padding) is list
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.pooling_type = pooling_type.upper()

        self.is_known_shape = True
        self.spatial_size = len(kernel_shape)
        self.input_rank = self.spatial_size + 2

        # if the rank is not defined, set it to the calculated input_rank
        # rank should be known for ops like tf.gather_nd
        # if not input.shape.rank:
        #   input.set_shape([None] * self.input_rank)
        self.orig_input_shape = flow_shape(input)
        self.input_shape = self.orig_input_shape

        if pooling_type.startswith("MAX"):
            # TODO(daquexian): change it
            self.padding_constant = -1.0e5
        else:
            self.padding_constant = 0

    def _calc_input_ind(self, output_ind, kernel, dilation, stride):
        """
            This function maps index from the output of _remove_dilations
            to index from the original input along single axis. It calculates
            the index inside the input data from the index of the output.
            It is used to generate the correct indexes of the values to be
            extracted by gather_nd.

            Args:
                output_ind: vector with indices from the output to be mapped
                kernel:     kernel size along the axis
                dilation:   dilations along the axis
                stride:     strides along the axis
            Return:
                input_ind: calculated indices

            The formula is:
                input_ind = (output_ind // kernel) * stride +
                            (output_ind % kernel) * dilation

            Example:
              If we have following 2D input to _remove_dilations:
                         [[  0,  1,  2,  3],
                          [  4,  5,  6,  7],
                          [  8,  9, 10, 11],
                          [ 12, 13, 14, 15]]
              and Kernel = [2, 2], Dilations: [2, 2], Strides: [1, 1]

              the output of _remove_dilations will have shape [4, 4] and
              _calc_input_ind will be called twice for the two axis 0 (along
              height) and axis 1 (along width) with

                  output_ind = [0, 1, 2, 3]

              which will result in:

                  input_ind = [0, 2, 1, 3]
        """
        return (output_ind // kernel) * (
            stride - kernel * dilation
        ) + output_ind * dilation

    def _calc_orig_argmax(self, ind):
        """
            Map result argxmax to the original input indices

            Maps indices generated by maxpool_with_argmax on top of the
            dilation reduced input to the orignal input indices
        """

        in_width = self.orig_input_shape[2]
        num_channels = self.orig_input_shape[3]
        output_width = self.output_shape[2]

        # mod_floor op is not implemented on GPU
        # implement it using: a % b = a - (a // b) * b

        # inRow = (ind // num_channels) // output_width
        # inCol = (ind // num_channels) % output_width
        # ind_channel = ind % num_channels

        ind_nhw = ind // num_channels

        inRow = ind_nhw // output_width
        inCol = ind_nhw - (ind_nhw // output_width) * output_width

        ind_channel = ind - ind_nhw * num_channels

        row = (
            self._calc_input_ind(
                inRow, self.kernel_shape[0], self.dilations[0], self.strides[0]
            )
            - self.pads[0]
        )
        col = (
            self._calc_input_ind(
                inCol, self.kernel_shape[1], self.dilations[1], self.strides[1]
            )
            - self.pads[2]
        )

        new_ind = num_channels * (row * in_width + col) + ind_channel
        return new_ind

    def _remove_dilations(self):
        """
            This method removes the dilations by extracting the values from
            the input for every sliding window according to the dilations,
            strides and kernel size and generates output that can be used by
            pooling operations with strides = kernel_shape to accomplish
            dilated pooling

            Example:
              Input:     [[  0,  1,  2,  3],
                          [  4,  5,  6,  7],
                          [  8,  9, 10, 11],
                          [ 12, 13, 14, 15]]

              Kernel:    [2, 2]
              Dilations: [2, 2]
              Strides:   [1, 1]

              Will return:
                         [[  0,  2,  1,  3],
                          [  8, 10,  9, 11],
                          [  4,  6,  5,  7],
                          [ 12, 14, 13, 15]]

              After max_pool2d with kernel_shape = strides = [2, 2]
              the result is:
                         [[ 10, 11],
                          [ 14, 15]]
        """

        input_shape = tf_shape(self.input)
        in_spatial_shape = input_shape[1 : self.spatial_size + 1]

        channels_count = input_shape[self.spatial_size + 1]
        # Initialize gather_ind with the range of channels
        # e.g. [0 1]
        gather_ind = tf.range(channels_count, dtype=tf.int64)
        # convert the vector to column vector
        # in the following logic we use column vectors
        gather_ind = tf.expand_dims(gather_ind, 1)

        # initilize the output_shape with zeros
        # self.output_shape will contain the shape of the
        # output tensor after the loop below is executed
        self.output_shape = [0] * (self.spatial_size + 2)
        self.output_shape[0] = input_shape[0]
        """
            Loop over the input spatial dimensions starting from the
            last (most internal) going up to the first dimension

            On every step of the loop calculate the output indices and
            map them to the input indices using `_calc_input_ind`,
            then "combine" with the already calculated indices from the
            previous dimensions using cartesian product.

            For the following example input:

              Input:     [[  0,  1,  2,  3],
                          [  4,  5,  6,  7],
                          [  8,  9, 10, 11],
                          [ 12, 13, 14, 15]]

              Kernel:    [2, 2]
              Dilations: [2, 2]
              Strides:   [1, 1]

            these are the steps that will be executed:

            1. Initilize gather_ind = [[0]]     # we have only 1 channel

            2. Loop step 0 (axis 1):
                  filter_size = 3
                  output_size = 4
                  dim_ind = [[0]
                             [2]
                             [1]
                             [3]]

                  gather_ind = [[0 0]
                                [2 0]
                                [1 0]
                                [3 0]]

            3. Loop step 1 (axis 0):
                  filter_size = 3
                  output_size = 4
                  dim_ind = [[0]
                             [2]
                             [1]
                             [3]]

                  gather_ind = [[0 0 0]
                                [0 2 0]
                                [0 1 0]
                                [0 3 0]
                                [2 0 0]
                                [2 2 0]
                                [2 1 0]
                                [2 3 0]
                                [1 0 0]
                                [1 2 0]
                                [1 1 0]
                                [1 3 0]
                                [3 0 0]
                                [3 2 0]
                                [3 1 0]
                                [3 3 0]]

            These are the indices used for gather_nd operation to collect
            the values from the input data.
        """

        for dim in range(self.spatial_size - 1, -1, -1):
            filter_size = (self.kernel_shape[dim] - 1) * self.dilations[dim] + 1
            output_size = (
                ((in_spatial_shape[dim] - filter_size) // self.strides[dim]) + 1
            ) * self.kernel_shape[dim]
            self.output_shape[dim + 1] = output_size

            # initialize the output dimension index with the range of the
            # dimension output size (e.g. 4): [0, 1, 2, 3]
            dim_ind = tf.range(output_size)

            # calculate the matching indices in the input data
            # [0, 1, 2, 3] will calculate to [0, 2, 1, 3]
            # from the above example
            dim_ind = self._calc_input_ind(
                dim_ind, self.kernel_shape[dim], self.dilations[dim], self.strides[dim]
            )
            # convert to column vector
            dim_ind = tf.expand_dims(dim_ind, 1)

            # "combine" current dimension indices with the previous dimensions
            # using cartesian product
            gather_ind = tf_product(dim_ind, gather_ind)

        # The result from the above loop for 2D data will be:
        # [[y1, x1, c], [y2, x2, c], ..., [yn, xm, c]] where n is the height,
        # m is the width and c is the channel number.

        # set the channels count in the output_shape
        self.output_shape[self.spatial_size + 1] = channels_count

        # expand the dimensions to match the input dimensions + 1
        for x in range(self.spatial_size):
            gather_ind = tf.expand_dims(gather_ind, 0)
        # dublicate the indices for every batch
        gather_ind = tf.tile(
            gather_ind, [input_shape[0]] + [1] * (self.spatial_size + 1)
        )

        # extract the selected values from the input
        output = tf.gather_nd(self.input, gather_ind, batch_dims=1)
        # reshape the output to the correct shape calculated earlier
        output = tf.reshape(output, self.output_shape)

        return output

    def _calc_pads_same(self, in_spatial_shape):
        """
            Calculate SAME_* paddings.
        """

        pad_ops = (
            pooling_helper.pad_numpy_ops
            if self.is_known_shape
            else pooling_helper.pad_tf_ops
        )

        return pooling_helper.calc_pads_same(
            in_spatial_shape,
            self.kernel_shape,
            self.strides,
            self.dilations,
            self.padding,
            pad_ops,
            2,
        )

    def _calc_pads_explicit(self):
        """
            Calculate explicit padding
        """
        assert type(self.padding) is list

        pads = []
        for i in range(self.spatial_size):
            pads += [self.padding[i], self.padding[i + self.spatial_size]]
        return pads

    def _calc_pads_ceil_mode(self, in_spatial_shape):
        """
            Calculate padding in ceil_mode
        """

        pads = []
        for i in range(self.spatial_size):
            dim_size = in_spatial_shape[i]
            filter_size = (self.kernel_shape[i] - 1) * self.dilations[i] + 1
            out_size = (dim_size - filter_size) / self.strides[i]
            if self.is_known_shape:
                pad_size = (np.ceil(out_size) - np.floor(out_size)).astype(np.int64)
            else:
                pad_size = tf.cast(
                    tf.math.ceil(out_size) - tf.math.floor(out_size), tf.int64
                )

            pads += [0, pad_size * self.strides[i]]
        return pads

    def _calc_pads(self, in_spatial_shape):
        if self.is_known_shape:
            pads = np.zeros([self.spatial_size * 2], np.int64)
        else:
            pads = tf.zeros([self.spatial_size * 2], tf.int64)

        # check for explicit padding
        if type(self.padding) is list:
            pads += self._calc_pads_explicit()
        elif self.padding.lower().startswith("same"):
            pads += self._calc_pads_same(in_spatial_shape)

        # when padding is set to SAME, ceil_mode will not do anything
        # because output sizes will be multiple of the strides
        if self.ceil_mode and (
            type(self.padding) is list or not self.padding.lower().startswith("same")
        ):
            new_spatial_shape = [
                in_spatial_shape[i] + pads[i * 2] + pads[i * 2 + 1]
                for i in range(self.spatial_size)
            ]
            pads += self._calc_pads_ceil_mode(new_spatial_shape)
        return pads

    def _pad_input(self):
        """
            Pad the input according to the parameters
        """
        # check if we need to do any padding at all
        if not self.ceil_mode and (
            (type(self.padding) is list and self.padding == [0] * self.spatial_size * 2)
            or self.padding == "VALID"
        ):
            self.pads = np.array([0] * self.spatial_size * 2)
            return (self.input, self.pads)

        in_spatial_shape = self.input_shape[-self.spatial_size :]
        pads = self._calc_pads(in_spatial_shape).tolist()

        if self.is_known_shape and np.count_nonzero(pads) == 0:
            self.pads = pads
            return (self.input, pads)

        tf_paddings = [[0, 0], [0, 0]]
        for i in range(self.spatial_size):
            tf_paddings += [[pads[i * 2], pads[i * 2 + 1]]]

        self.input = pad.pad(
            self.input, tf_paddings, constant_value=self.padding_constant
        )
        # update input shape and pads values
        self.input_shape = flow_shape(self.input)
        self.pads = pads

    def _calc_argmax_without_padding(self, ind):
        """
            Calculate the original indices as they would be without padding
        """
        in_width = self.orig_input_shape[2]
        padded_width = self.input_shape[2]
        num_channels = self.input_shape[3]

        # mod_floor op is not implemented on GPU
        # implement it using: a % b = a - (a // b) * b

        # ind_nhw = ind // num_channels
        # ind_channel = ind % num_channels

        ind_nhw = ind // num_channels
        ind_channel = ind - ind_nhw * num_channels

        new_ind = (ind_nhw // padded_width) * (self.pads[2] + self.pads[3])
        new_ind = ind_nhw - new_ind - self.pads[0] * in_width - self.pads[2]
        new_ind = num_channels * new_ind + ind_channel
        return new_ind

    def dilated_maxpool_with_argmax(self, force_custom_impl=False):
        """
            Do a dilated maxpool and return indices/argmax
        """
        # Tensorflow does not support maxpool_with_argmax on
        # spatial_size != 2
        assert self.spatial_size == 2

        if list(self.dilations) != [1] * self.spatial_size or force_custom_impl:
            # pad the input
            self._pad_input()

            new_input = self._remove_dilations()
            kernel_shape = [1] + list(self.kernel_shape) + [1]
            pooled, new_ind = tf.nn.max_pool_with_argmax(
                new_input, ksize=kernel_shape, strides=kernel_shape, padding="VALID"
            )
            new_ind = self._calc_orig_argmax(new_ind)
        else:
            self.pads = np.array([0] * self.spatial_size * 2)
            if type(self.padding) is list or self.padding.lower() == "same_lower":
                # pad the input
                self._pad_input()

                padding_ = "VALID"
            elif self.padding.lower() == "same_upper":
                padding_ = "SAME"
            else:
                padding_ = self.padding

            strides = [1] + list(self.strides) + [1]
            kernel_shape = [1] + list(self.kernel_shape) + [1]
            pooled, new_ind = tf.nn.max_pool_with_argmax(
                self.input, ksize=kernel_shape, strides=strides, padding=padding_
            )
            # if there was padding, recalculate the returned index
            # to exclude the padding
            if np.count_nonzero(self.pads) != 0:
                new_ind = self._calc_argmax_without_padding(new_ind)

        return (pooled, new_ind)

    def dilated_pool(self, force_custom_impl=False):
        """
            Does N-D dilated max/avg pooling. Pads the input if explicit or
            SAME_* padding is provided or ceil_mode is True
        """

        assert self.is_supported()

        if (
            self.is_explicit_padding
            or self.padding.lower() == "same_lower"
            or (self.padding.lower() == "same_upper" and self.count_include_pad)
        ):
            # pad the input
            self._pad_input()

            padding_ = "VALID"
        elif self.padding.lower() == "same_upper":
            padding_ = "SAME"
        else:
            padding_ = self.padding

        # if maxpool op with dialtions != 1 and spatial_size == 2
        # we can use tf.nn.dilation2d directly
        if (
            self.spatial_size == 2
            and self.pooling_type.startswith("MAX")
            and self.dilations != [1] * self.spatial_size
            and not force_custom_impl
        ):
            strides = [1] + list(self.strides) + [1]
            dilations = [1] + list(self.dilations) + [1]

            filter = tf.zeros(
                [self.kernel_shape[0], self.kernel_shape[1], self.input_shape[3]],
                self.input.dtype,
            )
            pooled = tf.nn.dilation2d(
                input=self.input,
                filters=filter,
                strides=strides,
                dilations=dilations,
                padding=padding_,
                data_format="NHWC",
            )
        # if spatial_size < 4 and strides == 1 or dilation == 1 use tf.nn.pool
        elif (
            self.spatial_size < 4
            and (
                self.strides == [1] * self.spatial_size
                or self.dilations == [1] * self.spatial_size
            )
            and not force_custom_impl
        ):
            # othwerwise check the pooling_type and use the correct op
            if self.pooling_type.startswith("MAX"):
                op = nn_ops.max_pool2d
            elif self.pooling_type == "AVG":
                op = nn_ops.avg_pool2d
            else:
                raise ValueError(
                    "%d-D %s pooling is not supported."
                    % (self.spatial_size, self.pooling_type)
                )
            pooled = op(
                self.input,
                ksize=self.kernel_shape,
                strides=self.strides,
                padding=padding_,
                data_format="NCHW",
            )
        # in any other case we use custom implementation _remove_dilations
        # to reduce atrous/dilated pooling into regular pooling and selecting
        # only the values of the input that should have been selected by
        # applying the strides and dilations. Then use tf.nn.pool with
        # strides = kernel_shape and no dilations
        else:
            if padding_ == "SAME":
                # pad the input
                self._pad_input()
            input_ = self._remove_dilations()
            pooled = tf.nn.pool(
                input_,
                window_shape=self.kernel_shape,
                strides=self.kernel_shape,
                padding="VALID",
                pooling_type=self.pooling_type,
            )
        return pooled

    def is_supported(self):
        """
            Function to check if the current set of arguments are
            supported for average pool
        """
        # check for maxpool
        if self.pooling_type.startswith("MAX"):
            return True
        else:
            # if count_include_pad is true it is fully supported
            if self.count_include_pad:
                return True
            # ceil mode is not supported
            elif self.ceil_mode:
                return False
            # explicit padding with padding values set to 0 is supported
            elif (
                self.is_explicit_padding and self.padding == [0] * self.spatial_size * 2
            ):
                return True
            # "valid" and "same_upper" auto padding is supported
            elif not self.is_explicit_padding and self.padding.lower() in [
                "valid",
                "same_upper",
            ]:
                return True
            # any other case is not supported
            else:
                return False
