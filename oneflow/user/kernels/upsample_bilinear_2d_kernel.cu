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
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/common/nd_index_offset_helper.h"
#include "oneflow/core/cuda/atomic.cuh"
#include "oneflow/user/kernels/upsample_kernel.h"

namespace oneflow {

namespace {

__device__ __forceinline__ void GetBilinearParamHalf(const bool align_corners, const int64_t h,
                                                     const int64_t w, const int64_t in_height,
                                                     const int64_t in_width, const double scale_h,
                                                     const double scale_w,
                                                     BilinearParam<half>* params) {
  half h1r;
  if (align_corners) {
    h1r = static_cast<half>(scale_h * static_cast<double>(h));
  } else {
    h1r = h1r = static_cast<half>((static_cast<double>(h) + 0.5f) * scale_h - 0.5f);
    h1r = h1r < static_cast<half>(0.0) ? static_cast<half>(0.0) : h1r;
  }
  const int64_t h1 = int(h1r);
  const int64_t h1p = (h1 < in_height - 1) ? 1 : 0;

  half w1r;
  if (align_corners) {
    w1r = static_cast<half>(scale_w * static_cast<double>(w));
  } else {
    w1r = static_cast<half>((static_cast<double>(w) + 0.5f) * scale_w - 0.5f);
    w1r = w1r < static_cast<half>(0.0) ? static_cast<half>(0.0) : w1r;
  }
  const int64_t w1 = int(w1r);
  const int64_t w1p = (w1 < in_width - 1) ? 1 : 0;

  params->top_h_index = h1;
  params->bottom_h_index = h1 + h1p;
  params->h_lerp = h1r - static_cast<half>(h1 * 1.0);
  params->left_w_index = w1;
  params->right_w_index = w1 + w1p;
  params->w_lerp = w1r - static_cast<half>(w1 * 1.0);
}

template<typename T>
__global__ void UpsampleBilinear2DForward(const int64_t elem_cnt, const T* in_dptr,
                                          NdIndexOffsetHelper<int64_t, 4> in_helper,
                                          NdIndexOffsetHelper<int64_t, 4> out_helper,
                                          const int64_t in_height, const int64_t in_width,
                                          const T scale_h, const T scale_w,
                                          const bool align_corners, T* out_dptr) {
  CUDA_1D_KERNEL_LOOP(index, elem_cnt) {
    int64_t n, c, h, w;
    out_helper.OffsetToNdIndex(index, n, c, h, w);
    BilinearParam<T> params;
    GetBilinearParam(align_corners, h, w, in_height, in_width, scale_h, scale_w, &params);
    const int64_t top_offset = in_helper.NdIndexToOffset(n, c, params.top_h_index, 0);
    const int64_t bottom_offset = in_helper.NdIndexToOffset(n, c, params.bottom_h_index, 0);
    const T top_left = in_dptr[top_offset + params.left_w_index];
    const T top_right = in_dptr[top_offset + params.right_w_index];
    const T bottom_left = in_dptr[bottom_offset + params.left_w_index];
    const T bottom_right = in_dptr[bottom_offset + params.right_w_index];
    out_dptr[index] =
        (1 - params.h_lerp) * ((1 - params.w_lerp) * top_left + params.w_lerp * top_right)
        + params.h_lerp * ((1 - params.w_lerp) * bottom_left + params.w_lerp * bottom_right);
  }
}

template<>
__global__ void UpsampleBilinear2DForward(const int64_t elem_cnt, const half* in_dptr,
                                          NdIndexOffsetHelper<int64_t, 4> in_helper,
                                          NdIndexOffsetHelper<int64_t, 4> out_helper,
                                          const int64_t in_height, const int64_t in_width,
                                          const half scale_h, const half scale_w,
                                          const bool align_corners, half* out_dptr) {
  CUDA_1D_KERNEL_LOOP(index, elem_cnt) {
    int64_t n, c, h, w;
    out_helper.OffsetToNdIndex(index, n, c, h, w);
    BilinearParam<half> params;
    GetBilinearParamHalf(align_corners, h, w, in_height, in_width, scale_h, scale_w, &params);
    const int64_t top_offset = in_helper.NdIndexToOffset(n, c, params.top_h_index, 0);
    const int64_t bottom_offset = in_helper.NdIndexToOffset(n, c, params.bottom_h_index, 0);
    const half top_left = in_dptr[top_offset + params.left_w_index];
    const half top_right = in_dptr[top_offset + params.right_w_index];
    const half bottom_left = in_dptr[bottom_offset + params.left_w_index];
    const half bottom_right = in_dptr[bottom_offset + params.right_w_index];
    out_dptr[index] =
        (static_cast<half>(1.0) - params.h_lerp)
            * ((static_cast<half>(1.0) - params.w_lerp) * top_left + params.w_lerp * top_right)
        + params.h_lerp
              * ((static_cast<half>(1.0) - params.w_lerp) * bottom_left
                 + params.w_lerp * bottom_right);
  }
}

template<typename T>
__global__ void UpsampleBilinearBackward(const int64_t elem_cnt, const T* dy_dptr,
                                         NdIndexOffsetHelper<int64_t, 4> dy_helper,
                                         NdIndexOffsetHelper<int64_t, 4> dx_helper,
                                         const int64_t dx_height, const int64_t dx_width,
                                         const T scale_h, const T scale_w, const bool align_corners,
                                         T* dx_dptr) {
  CUDA_1D_KERNEL_LOOP(index, elem_cnt) {
    int64_t n, c, h, w;
    dy_helper.OffsetToNdIndex(index, n, c, h, w);
    BilinearParam<T> params;
    GetBilinearParam(align_corners, h, w, dx_height, dx_width, scale_h, scale_w, &params);
    const int64_t top_offset = dx_helper.NdIndexToOffset(n, c, params.top_h_index, 0);
    const int64_t bottom_offset = dx_helper.NdIndexToOffset(n, c, params.bottom_h_index, 0);
    const T dy = dy_dptr[index];
    const T dbottom = params.h_lerp * dy;
    T* dx_dptr_bottom_offset = dx_dptr + bottom_offset;
    cuda::atomic::FastAdd(dx_dptr_bottom_offset, params.left_w_index, elem_cnt,
                          static_cast<T>((1 - params.w_lerp) * dbottom));
    cuda::atomic::FastAdd(dx_dptr_bottom_offset, params.right_w_index, elem_cnt,
                          static_cast<T>(params.w_lerp * dbottom));
    const T dtop = dy - dbottom;
    T* dx_dptr_top_offset = dx_dptr + top_offset;
    cuda::atomic::FastAdd(dx_dptr_top_offset, params.left_w_index, elem_cnt,
                          static_cast<T>((1 - params.w_lerp) * dtop));
    cuda::atomic::FastAdd(dx_dptr_top_offset, params.right_w_index, elem_cnt,
                          static_cast<T>(params.w_lerp * dtop));
  }
}

template<>
__global__ void UpsampleBilinearBackward(const int64_t elem_cnt, const half* dy_dptr,
                                         NdIndexOffsetHelper<int64_t, 4> dy_helper,
                                         NdIndexOffsetHelper<int64_t, 4> dx_helper,
                                         const int64_t dx_height, const int64_t dx_width,
                                         const half scale_h, const half scale_w,
                                         const bool align_corners, half* dx_dptr) {
  CUDA_1D_KERNEL_LOOP(index, elem_cnt) {
    int64_t n, c, h, w;
    dy_helper.OffsetToNdIndex(index, n, c, h, w);
    BilinearParam<half> params;
    GetBilinearParamHalf(align_corners, h, w, dx_height, dx_width, scale_h, scale_w, &params);
    const int64_t top_offset = dx_helper.NdIndexToOffset(n, c, params.top_h_index, 0);
    const int64_t bottom_offset = dx_helper.NdIndexToOffset(n, c, params.bottom_h_index, 0);
    const half dy = dy_dptr[index];
    const half dbottom = params.h_lerp * dy;
    half* dx_dptr_bottom_offset = dx_dptr + bottom_offset;
    cuda::atomic::FastAdd(dx_dptr_bottom_offset, params.left_w_index, elem_cnt,
                          static_cast<half>((static_cast<half>(1.0) - params.w_lerp) * dbottom));
    cuda::atomic::FastAdd(dx_dptr_bottom_offset, params.right_w_index, elem_cnt,
                          static_cast<half>(params.w_lerp * dbottom));
    const half dtop = dy - dbottom;
    half* dx_dptr_top_offset = dx_dptr + top_offset;
    cuda::atomic::FastAdd(dx_dptr_top_offset, params.left_w_index, elem_cnt,
                          static_cast<half>((static_cast<half>(1.0) - params.w_lerp) * dtop));
    cuda::atomic::FastAdd(dx_dptr_top_offset, params.right_w_index, elem_cnt,
                          static_cast<half>(params.w_lerp * dtop));
  }
}

}  // namespace

template<typename T>
class UpsampleBilinear2DGPUKernel final : public user_op::OpKernel {
 public:
  UpsampleBilinear2DGPUKernel() = default;
  ~UpsampleBilinear2DGPUKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y_tensor = ctx->Tensor4ArgNameAndIndex("y", 0);
    const bool align_corners = ctx->Attr<bool>("align_corners");
    const std::vector<int64_t> output_size = ctx->Attr<std::vector<int64_t>>("output_size");
    double height_scale = ctx->Attr<double>("height_scale");
    double width_scale = ctx->Attr<double>("width_scale");
    const int64_t elem_cnt = y_tensor->shape_view().elem_cnt();
    NdIndexOffsetHelper<int64_t, 4> in_helper(
        x_tensor->shape_view().At(0), x_tensor->shape_view().At(1), x_tensor->shape_view().At(2),
        x_tensor->shape_view().At(3));
    NdIndexOffsetHelper<int64_t, 4> out_helper(
        y_tensor->shape_view().At(0), y_tensor->shape_view().At(1), y_tensor->shape_view().At(2),
        y_tensor->shape_view().At(3));

    const int64_t in_height = x_tensor->shape_view().At(2);
    const int64_t in_width = x_tensor->shape_view().At(3);
    const int64_t out_height = y_tensor->shape_view().At(2);
    const int64_t out_width = y_tensor->shape_view().At(3);
    if (!output_size.empty()) {
      height_scale = static_cast<double>(out_height) / static_cast<double>(in_height);
      width_scale = static_cast<double>(out_width) / static_cast<double>(in_width);
    }
    if (in_height == out_height && in_width == out_width) {
      Memcpy<DeviceType::kCUDA>(
          ctx->stream(), y_tensor->mut_dptr<void>(), x_tensor->dptr<void>(),
          x_tensor->shape_view().elem_cnt() * GetSizeOfDataType(x_tensor->data_type()));
    } else {
      const T scale_height = GetAreaPixelScale(in_height, out_height, align_corners, height_scale);
      const T scale_width = GetAreaPixelScale(in_width, out_width, align_corners, width_scale);
      RUN_CUDA_KERNEL((UpsampleBilinear2DForward<T>), ctx->stream(), elem_cnt, elem_cnt,
                      x_tensor->dptr<T>(), in_helper, out_helper, in_height, in_width, scale_height,
                      scale_width, align_corners, y_tensor->mut_dptr<T>());
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
class UpsampleBilinear2DGradGPUKernel final : public user_op::OpKernel {
 public:
  UpsampleBilinear2DGradGPUKernel() = default;
  ~UpsampleBilinear2DGradGPUKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* dx_tensor = ctx->Tensor4ArgNameAndIndex("dx", 0);
    Memset<DeviceType::kCUDA>(ctx->stream(), dx_tensor->mut_dptr<T>(), 0,
                              dx_tensor->shape_view().elem_cnt() * sizeof(T));
    const user_op::Tensor* dy_tensor = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const bool align_corners = ctx->Attr<bool>("align_corners");
    const std::vector<int64_t> output_size = ctx->Attr<std::vector<int64_t>>("output_size");
    double height_scale = ctx->Attr<double>("height_scale");
    double width_scale = ctx->Attr<double>("width_scale");
    const int64_t elem_cnt = dy_tensor->shape_view().elem_cnt();
    NdIndexOffsetHelper<int64_t, 4> dy_helper(
        dy_tensor->shape_view().At(0), dy_tensor->shape_view().At(1), dy_tensor->shape_view().At(2),
        dy_tensor->shape_view().At(3));
    NdIndexOffsetHelper<int64_t, 4> dx_helper(
        dx_tensor->shape_view().At(0), dx_tensor->shape_view().At(1), dx_tensor->shape_view().At(2),
        dx_tensor->shape_view().At(3));

    const int64_t in_height = dx_tensor->shape_view().At(2);
    const int64_t in_width = dx_tensor->shape_view().At(3);
    const int64_t out_height = dy_tensor->shape_view().At(2);
    const int64_t out_width = dy_tensor->shape_view().At(3);
    if (!output_size.empty()) {
      height_scale = static_cast<double>(out_height) / static_cast<double>(in_height);
      width_scale = static_cast<double>(out_width) / static_cast<double>(in_width);
    }
    if (in_height == out_height && in_width == out_width) {
      Memcpy<DeviceType::kCUDA>(
          ctx->stream(), dx_tensor->mut_dptr<void>(), dy_tensor->dptr<void>(),
          dy_tensor->shape_view().elem_cnt() * GetSizeOfDataType(dy_tensor->data_type()));
    } else {
      const T scale_height = GetAreaPixelScale(in_height, out_height, align_corners, height_scale);
      const T scale_width = GetAreaPixelScale(in_width, out_width, align_corners, width_scale);
      RUN_CUDA_KERNEL((UpsampleBilinearBackward<T>), ctx->stream(), elem_cnt, elem_cnt,
                      dy_tensor->dptr<T>(), dy_helper, dx_helper, in_height, in_width, scale_height,
                      scale_width, align_corners, dx_tensor->mut_dptr<T>());
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_UPSAMPLE_BILINEAR_2D_CUDA_KERNEL(dtype)                                \
  REGISTER_USER_KERNEL("upsample_bilinear_2d")                                          \
      .SetCreateFn<UpsampleBilinear2DGPUKernel<dtype>>()                                \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                  \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("upsample_bilinear_2d_grad")                                     \
      .SetCreateFn<UpsampleBilinear2DGradGPUKernel<dtype>>()                            \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                  \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_UPSAMPLE_BILINEAR_2D_CUDA_KERNEL(half)
REGISTER_UPSAMPLE_BILINEAR_2D_CUDA_KERNEL(float)
REGISTER_UPSAMPLE_BILINEAR_2D_CUDA_KERNEL(double)

}  // namespace oneflow
