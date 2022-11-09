/*
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
*/

#include "grid_sample_kernel_util.h"

namespace oneflow {

template<typename data_type, typename index_type>
struct GridSampleKernelUtil<DeviceType::kCPU, data_type, index_type> final {
  static void Forward4D(user_op::KernelComputeContext* ctx, const user_op::Tensor* input,
                        const user_op::Tensor* grid, user_op::Tensor* output,
                        GridSamplerInterpolation interpolation, GridSamplerPadding padding,
                        const bool align_corners, const ShapeView& input_shape,
                        const ShapeView& grid_shape, const ShapeView& output_shape, int64_t count) {
    GridSampler4DKernel<data_type, index_type>(
        count, input->dptr<data_type>(), grid->dptr<data_type>(), output->mut_dptr<data_type>(),
        input_shape.At(0), input_shape.At(1), input_shape.At(2), input_shape.At(3),
        output_shape.At(2), output_shape.At(3), interpolation, padding, align_corners);
  }

  static void Forward5D(user_op::KernelComputeContext* ctx, const user_op::Tensor* input,
                        const user_op::Tensor* grid, user_op::Tensor* output,
                        GridSamplerInterpolation interpolation, GridSamplerPadding padding,
                        const bool align_corners, const ShapeView& input_shape,
                        const ShapeView& grid_shape, const ShapeView& output_shape, int64_t count) {
    GridSampler5DKernel<data_type, index_type>(
        count, input->dptr<data_type>(), grid->dptr<data_type>(), output->mut_dptr<data_type>(),
        input_shape.At(0), input_shape.At(1), input_shape.At(2), input_shape.At(3),
        input_shape.At(4), output_shape.At(2), output_shape.At(3), output_shape.At(4),
        interpolation, padding, align_corners);
  }

  static void Backward4D(user_op::KernelComputeContext* ctx, const user_op::Tensor* doutput,
                         const user_op::Tensor* input, const user_op::Tensor* grid,
                         user_op::Tensor* dinput, user_op::Tensor* dgrid,
                         GridSamplerInterpolation interpolation, GridSamplerPadding padding,
                         const bool align_corners, const ShapeView& input_shape,
                         const ShapeView& grid_shape, const ShapeView& output_shape,
                         int64_t count) {
    GridSampler4DBackwardKernel<data_type, index_type>(
        count, doutput->dptr<data_type>(), input->dptr<data_type>(), grid->dptr<data_type>(),
        dinput->mut_dptr<data_type>(), dgrid->mut_dptr<data_type>(), input_shape.At(0),
        input_shape.At(1), input_shape.At(2), input_shape.At(3), output_shape.At(2),
        output_shape.At(3), interpolation, padding, align_corners, input_shape.elem_cnt());
  }

  static void Backward5D(user_op::KernelComputeContext* ctx, const user_op::Tensor* doutput,
                         const user_op::Tensor* input, const user_op::Tensor* grid,
                         user_op::Tensor* dinput, user_op::Tensor* dgrid,
                         GridSamplerInterpolation interpolation, GridSamplerPadding padding,
                         const bool align_corners, const ShapeView& input_shape,
                         const ShapeView& grid_shape, const ShapeView& output_shape,
                         int64_t count) {
    GridSampler5DBackwardKernel<data_type, index_type>(
        count, doutput->dptr<data_type>(), input->dptr<data_type>(), grid->dptr<data_type>(),
        dinput->mut_dptr<data_type>(), dgrid->mut_dptr<data_type>(), input_shape.At(0),
        input_shape.At(1), input_shape.At(2), input_shape.At(3), input_shape.At(4),
        output_shape.At(2), output_shape.At(3), output_shape.At(4), interpolation, padding,
        align_corners, input_shape.elem_cnt());
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_GRID_SAMPLE_KERNEL_UTIL, (DeviceType::kCPU),
                                 FLOATING_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ);

}  // namespace oneflow
