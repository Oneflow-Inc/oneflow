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

#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/framework/config_def.h"
#include "grid_sample_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename data_type>
class GridSampleKernel final : public user_op::OpKernel {
 public:
  GridSampleKernel() = default;
  ~GridSampleKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* input = ctx->Tensor4ArgNameAndIndex("input", 0);
    const user_op::Tensor* grid = ctx->Tensor4ArgNameAndIndex("grid", 0);
    user_op::Tensor* output = ctx->Tensor4ArgNameAndIndex("output", 0);
    const std::string interpolation_mode = ctx->Attr<std::string>("interpolation_mode");
    const std::string padding_mode = ctx->Attr<std::string>("padding_mode");
    GridSamplerInterpolation interpolation = StringToGridSamplerInterpolation(interpolation_mode);
    GridSamplerPadding padding = StringToGridGridSamplerPadding(padding_mode);
    const bool align_corners = ctx->Attr<bool>("align_corners");

    const ShapeView& input_shape = input->shape_view();
    const ShapeView& grid_shape = grid->shape_view();
    const ShapeView& output_shape = output->shape_view();
    int64_t count = output_shape.elem_cnt() / input_shape.At(1);

    if (input_shape.NumAxes() == 4) {
      if (!CanUse32BitIndex({input_shape, grid_shape, output_shape})) {
        GridSampleKernelUtil<device_type, data_type, int64_t>::Forward4D(
            ctx, input, grid, output, interpolation, padding, align_corners, input_shape,
            grid_shape, output_shape, count);
      } else {
        GridSampleKernelUtil<device_type, data_type, int32_t>::Forward4D(
            ctx, input, grid, output, interpolation, padding, align_corners, input_shape,
            grid_shape, output_shape, count);
      }
    } else {
      if (!CanUse32BitIndex({input_shape, grid_shape, output_shape})) {
        GridSampleKernelUtil<device_type, data_type, int64_t>::Forward5D(
            ctx, input, grid, output, interpolation, padding, align_corners, input_shape,
            grid_shape, output_shape, count);
      } else {
        GridSampleKernelUtil<device_type, data_type, int32_t>::Forward5D(
            ctx, input, grid, output, interpolation, padding, align_corners, input_shape,
            grid_shape, output_shape, count);
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GRID_SAMPLE_KERNEL(device, dtype)          \
  REGISTER_USER_KERNEL("grid_sample")                       \
      .SetCreateFn<GridSampleKernel<device, dtype>>()       \
      .SetIsMatchedHob((user_op::HobDeviceType() == device) \
                       && (user_op::HobDataType("input", 0) == GetDataType<dtype>::value))

REGISTER_GRID_SAMPLE_KERNEL(DeviceType::kCPU, float);
REGISTER_GRID_SAMPLE_KERNEL(DeviceType::kCPU, double);
#ifdef WITH_CUDA
REGISTER_GRID_SAMPLE_KERNEL(DeviceType::kCUDA, float);
REGISTER_GRID_SAMPLE_KERNEL(DeviceType::kCUDA, double);
#endif

template<DeviceType device_type, typename data_type>
class GridSampleGradKernel final : public user_op::OpKernel {
 public:
  GridSampleGradKernel() = default;
  ~GridSampleGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* doutput = ctx->Tensor4ArgNameAndIndex("doutput", 0);
    const user_op::Tensor* input = ctx->Tensor4ArgNameAndIndex("input", 0);
    const user_op::Tensor* grid = ctx->Tensor4ArgNameAndIndex("grid", 0);
    user_op::Tensor* dinput = ctx->Tensor4ArgNameAndIndex("dinput", 0);
    user_op::Tensor* dgrid = ctx->Tensor4ArgNameAndIndex("dgrid", 0);
    const std::string interpolation_mode = ctx->Attr<std::string>("interpolation_mode");
    const std::string padding_mode = ctx->Attr<std::string>("padding_mode");
    GridSamplerInterpolation interpolation = StringToGridSamplerInterpolation(interpolation_mode);
    GridSamplerPadding padding = StringToGridGridSamplerPadding(padding_mode);
    const bool align_corners = ctx->Attr<bool>("align_corners");

    const ShapeView& input_shape = input->shape_view();
    const ShapeView& grid_shape = grid->shape_view();
    const ShapeView& output_shape = doutput->shape_view();
    int64_t count = output_shape.elem_cnt() / input_shape.At(1);

    Memset<device_type>(ctx->stream(), dinput->mut_dptr<data_type>(), 0,
                        input_shape.elem_cnt() * sizeof(data_type));

    if (input_shape.NumAxes() == 4) {
      if (!CanUse32BitIndex({input_shape, grid_shape, output_shape})) {
        GridSampleKernelUtil<device_type, data_type, int64_t>::Backward4D(
            ctx, doutput, input, grid, dinput, dgrid, interpolation, padding, align_corners,
            input_shape, grid_shape, output_shape, count);
      } else {
        GridSampleKernelUtil<device_type, data_type, int32_t>::Backward4D(
            ctx, doutput, input, grid, dinput, dgrid, interpolation, padding, align_corners,
            input_shape, grid_shape, output_shape, count);
      }
    } else {
      if (!CanUse32BitIndex({input_shape, grid_shape, output_shape})) {
        GridSampleKernelUtil<device_type, data_type, int64_t>::Backward5D(
            ctx, doutput, input, grid, dinput, dgrid, interpolation, padding, align_corners,
            input_shape, grid_shape, output_shape, count);
      } else {
        GridSampleKernelUtil<device_type, data_type, int32_t>::Backward5D(
            ctx, doutput, input, grid, dinput, dgrid, interpolation, padding, align_corners,
            input_shape, grid_shape, output_shape, count);
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GRID_SAMPLE_GRAD_KERNEL(device, dtype)     \
  REGISTER_USER_KERNEL("grid_sample_grad")                  \
      .SetCreateFn<GridSampleGradKernel<device, dtype>>()   \
      .SetIsMatchedHob((user_op::HobDeviceType() == device) \
                       && (user_op::HobDataType("input", 0) == GetDataType<dtype>::value))

REGISTER_GRID_SAMPLE_GRAD_KERNEL(DeviceType::kCPU, float);
REGISTER_GRID_SAMPLE_GRAD_KERNEL(DeviceType::kCPU, double);
#ifdef WITH_CUDA
REGISTER_GRID_SAMPLE_GRAD_KERNEL(DeviceType::kCUDA, float);
REGISTER_GRID_SAMPLE_GRAD_KERNEL(DeviceType::kCUDA, double);
#endif

}  // namespace oneflow
