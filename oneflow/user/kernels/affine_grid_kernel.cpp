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
#include "affine_grid_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename data_type>
class AffineGridKernel final : public user_op::OpKernel {
 public:
  AffineGridKernel() = default;
  ~AffineGridKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* theta = ctx->Tensor4ArgNameAndIndex("theta", 0);
    user_op::Tensor* grid = ctx->Tensor4ArgNameAndIndex("grid", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const Shape& size = ctx->Attr<Shape>("size");
    const bool& align_corners = ctx->Attr<bool>("align_corners");
    bool is_2d_grid = true;
    if (size.NumAxes() == 5) { is_2d_grid = false; }

    int64_t N = theta->shape_view().At(0);
    int64_t theta_h = theta->shape_view().At(1);
    int64_t theta_w = theta->shape_view().At(2);

    if (is_2d_grid) {
      int64_t H = size.At(2);
      int64_t W = size.At(3);
      // generate base grid
      GenerateBaseGridImp<device_type>::Generate2D(ctx, tmp_buffer->mut_dptr<data_type>(), H, W,
                                                   align_corners);
      // Compute each batch
      for (int n = 0; n < N; n++) {
        NewKernelUtil<device_type>::OFGemm(ctx->stream(), CblasNoTrans, CblasTrans, H * W, theta_h,
                                           theta_w, 1.0, tmp_buffer->dptr<data_type>(),
                                           theta->dptr<data_type>() + n * theta_h * theta_w, 0.0,
                                           grid->mut_dptr<data_type>() + n * theta_h * H * W);
      }
    } else {
      int64_t D = size.At(2);
      int64_t H = size.At(3);
      int64_t W = size.At(4);
      // generate base grid
      GenerateBaseGridImp<device_type>::Generate3D(ctx, tmp_buffer->mut_dptr<data_type>(), D, H, W,
                                                   align_corners);
      // Compute each batch
      for (int n = 0; n < N; n++) {
        NewKernelUtil<device_type>::OFGemm(ctx->stream(), CblasNoTrans, CblasTrans, D * H * W,
                                           theta_h, theta_w, 1.0, tmp_buffer->dptr<data_type>(),
                                           theta->dptr<data_type>() + n * theta_h * theta_w, 0.0,
                                           grid->mut_dptr<data_type>() + n * theta_h * D * H * W);
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_AFFINE_GRID_KERNEL(device, dtype)                                         \
  REGISTER_USER_KERNEL("affine_grid")                                                      \
      .SetCreateFn<AffineGridKernel<device, dtype>>()                                      \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                                \
                       && (user_op::HobDataType("theta", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t {                        \
        const Shape& size = ctx->Attr<Shape>("size");                                      \
        size_t tmp_buffer_size = size.Count(2) * (size.NumAxes() - 1) * sizeof(dtype);     \
        return tmp_buffer_size;                                                            \
      })

REGISTER_AFFINE_GRID_KERNEL(DeviceType::kCPU, float);
REGISTER_AFFINE_GRID_KERNEL(DeviceType::kCPU, double);
#ifdef WITH_CUDA
REGISTER_AFFINE_GRID_KERNEL(DeviceType::kCUDA, float);
REGISTER_AFFINE_GRID_KERNEL(DeviceType::kCUDA, double);
#endif

template<DeviceType device_type, typename data_type>
class AffineGridGradKernel final : public user_op::OpKernel {
 public:
  AffineGridGradKernel() = default;
  ~AffineGridGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dgrid = ctx->Tensor4ArgNameAndIndex("dgrid", 0);
    user_op::Tensor* dtheta = ctx->Tensor4ArgNameAndIndex("dtheta", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const Shape& size = ctx->Attr<Shape>("size");
    const bool& align_corners = ctx->Attr<bool>("align_corners");
    bool is_2d_grid = true;
    if (size.NumAxes() == 5) { is_2d_grid = false; }

    int64_t N = dtheta->shape_view().At(0);
    int64_t dtheta_h = dtheta->shape_view().At(1);
    int64_t dtheta_w = dtheta->shape_view().At(2);

    if (is_2d_grid) {
      int64_t H = size.At(2);
      int64_t W = size.At(3);
      // generate base grid
      GenerateBaseGridImp<device_type>::Generate2D(ctx, tmp_buffer->mut_dptr<data_type>(), H, W,
                                                   align_corners);
      // Compute each batch
      for (int n = 0; n < N; n++) {
        NewKernelUtil<device_type>::OFGemm(
            ctx->stream(), CblasTrans, CblasNoTrans, dtheta_h, dtheta_w, H * W, 1.0,
            dgrid->dptr<data_type>() + n * dtheta_h * H * W, tmp_buffer->dptr<data_type>(), 0.0,
            dtheta->mut_dptr<data_type>() + n * dtheta_h * dtheta_w);
      }
    } else {
      int64_t D = size.At(2);
      int64_t H = size.At(3);
      int64_t W = size.At(4);
      GenerateBaseGridImp<device_type>::Generate3D(ctx, tmp_buffer->mut_dptr<data_type>(), D, H, W,
                                                   align_corners);
      // Compute each batch
      for (int n = 0; n < N; n++) {
        NewKernelUtil<device_type>::OFGemm(
            ctx->stream(), CblasTrans, CblasNoTrans, dtheta_h, dtheta_w, D * H * W, 1.0,
            dgrid->dptr<data_type>() + n * dtheta_h * D * H * W, tmp_buffer->dptr<data_type>(), 0.0,
            dtheta->mut_dptr<data_type>() + n * dtheta_h * dtheta_w);
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_AFFINE_GRID_GRAD_KERNEL(device, dtype)                                    \
  REGISTER_USER_KERNEL("affine_grid_grad")                                                 \
      .SetCreateFn<AffineGridGradKernel<device, dtype>>()                                  \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                                \
                       && (user_op::HobDataType("dgrid", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t {                        \
        const Shape& size = ctx->Attr<Shape>("size");                                      \
        size_t tmp_buffer_size = size.Count(2) * (size.NumAxes() - 1) * sizeof(dtype);     \
        return tmp_buffer_size;                                                            \
      })

REGISTER_AFFINE_GRID_GRAD_KERNEL(DeviceType::kCPU, float);
REGISTER_AFFINE_GRID_GRAD_KERNEL(DeviceType::kCPU, double);
#ifdef WITH_CUDA
REGISTER_AFFINE_GRID_GRAD_KERNEL(DeviceType::kCUDA, float);
REGISTER_AFFINE_GRID_GRAD_KERNEL(DeviceType::kCUDA, double);
#endif

}  // namespace oneflow
