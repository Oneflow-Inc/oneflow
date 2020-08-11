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

namespace oneflow {

namespace {

__device__ int64_t GetNearestInputIndex(const int64_t out_dim_idx, const float scale,
                                        const int64_t in_dim_size) {
  return max(min(static_cast<int64_t>(floorf((static_cast<float>(out_dim_idx) + 0.5f) * scale)),
                 in_dim_size - 1),
             static_cast<int64_t>(0));
}

template<typename T>
__global__ void UpsampleNearestForward(const int64_t elem_cnt, const T* in_dptr,
                                       NdIndexOffsetHelper<int64_t, 4> in_helper,
                                       NdIndexOffsetHelper<int64_t, 4> out_helper,
                                       const int64_t in_height, const int64_t in_width,
                                       const float scale_h, const float scale_w, T* out_dptr) {
  CUDA_1D_KERNEL_LOOP(index, elem_cnt) {
    int64_t n, c, h, w;
    out_helper.OffsetToNdIndex(index, n, c, h, w);
    const int64_t in_h = GetNearestInputIndex(h, scale_h, in_height);
    const int64_t in_w = GetNearestInputIndex(w, scale_w, in_width);
    out_dptr[index] = in_dptr[in_helper.NdIndexToOffset(n, c, in_h, in_w)];
  }
}

template<typename T>
__global__ void UpsampleNearestBackward(const int64_t elem_cnt, const T* dy_dptr,
                                        NdIndexOffsetHelper<int64_t, 4> dy_helper,
                                        NdIndexOffsetHelper<int64_t, 4> dx_helper,
                                        const int64_t dx_height, const int64_t dx_width,
                                        const float scale_h, const float scale_w, T* dx_dptr) {
  CUDA_1D_KERNEL_LOOP(index, elem_cnt) {
    int64_t n, c, h, w;
    dy_helper.OffsetToNdIndex(index, n, c, h, w);
    const int64_t dx_h = GetNearestInputIndex(h, scale_h, dx_height);
    const int64_t dx_w = GetNearestInputIndex(w, scale_w, dx_width);
    atomicAdd(dx_dptr + dx_helper.NdIndexToOffset(n, c, dx_h, dx_w), dy_dptr[index]);
  }
}

struct BilinearParam {
  int64_t top_h_index;
  int64_t bottom_h_index;
  int64_t left_w_index;
  int64_t right_w_index;
  float w_lerp;
  float h_lerp;
};

__device__ void GetBilinearParam(const int64_t index, const int64_t h, const int64_t w,
                                 const int64_t in_height, const int64_t in_width,
                                 const float scale_h, const float scale_w, BilinearParam* params) {
  const float in_h = (static_cast<float>(h) + 0.5f) * scale_h - 0.5f;
  const float in_w = (static_cast<float>(w) + 0.5f) * scale_w - 0.5f;
  params->top_h_index = in_h > 0.0 ? floorf(in_h) : 0;
  params->bottom_h_index = (in_h < in_height - 1) ? ceilf(in_h) : in_height - 1;
  params->h_lerp = in_h - floorf(in_h);
  params->left_w_index = in_w > 0.0 ? floorf(in_w) : 0;
  params->right_w_index = (in_w < in_width - 1) ? ceilf(in_w) : in_width - 1;
  params->w_lerp = in_w - floorf(in_w);
}

template<typename T>
__global__ void UpsampleBilinearForward(const int64_t elem_cnt, const T* in_dptr,
                                        NdIndexOffsetHelper<int64_t, 4> in_helper,
                                        NdIndexOffsetHelper<int64_t, 4> out_helper,
                                        const int64_t in_height, const int64_t in_width,
                                        const float scale_h, const float scale_w, T* out_dptr) {
  CUDA_1D_KERNEL_LOOP(index, elem_cnt) {
    int64_t n, c, h, w;
    out_helper.OffsetToNdIndex(index, n, c, h, w);
    BilinearParam params;
    GetBilinearParam(index, h, w, in_height, in_width, scale_h, scale_w, &params);
    const int64_t top_offset = in_helper.NdIndexToOffset(n, c, params.top_h_index, 0);
    const int64_t bottom_offset = in_helper.NdIndexToOffset(n, c, params.bottom_h_index, 0);
    const float top_left = in_dptr[top_offset + params.left_w_index];
    const float top_right = in_dptr[top_offset + params.right_w_index];
    const float bottom_left = in_dptr[bottom_offset + params.left_w_index];
    const float bottom_right = in_dptr[bottom_offset + params.right_w_index];
    const float top = top_left + (top_right - top_left) * params.w_lerp;
    const float bottom = bottom_left + (bottom_right - bottom_left) * params.w_lerp;
    out_dptr[index] = top + (bottom - top) * params.h_lerp;
  }
}

template<typename T>
__global__ void UpsampleBilinearBackward(const int64_t elem_cnt, const T* dy_dptr,
                                         NdIndexOffsetHelper<int64_t, 4> dy_helper,
                                         NdIndexOffsetHelper<int64_t, 4> dx_helper,
                                         const int64_t dx_height, const int64_t dx_width,
                                         const float scale_h, const float scale_w, T* dx_dptr) {
  CUDA_1D_KERNEL_LOOP(index, elem_cnt) {
    int64_t n, c, h, w;
    dy_helper.OffsetToNdIndex(index, n, c, h, w);
    BilinearParam params;
    GetBilinearParam(index, h, w, dx_height, dx_width, scale_h, scale_w, &params);
    const int64_t top_offset = dx_helper.NdIndexToOffset(n, c, params.top_h_index, 0);
    const int64_t bottom_offset = dx_helper.NdIndexToOffset(n, c, params.bottom_h_index, 0);
    const T dy = dy_dptr[index];
    const float dbottom = params.h_lerp * dy;
    T* dx_dptr_bottom_offset = dx_dptr + bottom_offset;
    atomicAdd(dx_dptr_bottom_offset + params.left_w_index,
              static_cast<T>((1 - params.w_lerp) * dbottom));
    atomicAdd(dx_dptr_bottom_offset + params.right_w_index,
              static_cast<T>(params.w_lerp * dbottom));
    const float dtop = dy - dbottom;
    T* dx_dptr_top_offset = dx_dptr + top_offset;
    atomicAdd(dx_dptr_top_offset + params.left_w_index, static_cast<T>((1 - params.w_lerp) * dtop));
    atomicAdd(dx_dptr_top_offset + params.right_w_index, static_cast<T>(params.w_lerp * dtop));
  }
}

}  // namespace

template<typename T>
class UpsampleNearestGPUKernel final : public user_op::OpKernel {
 public:
  UpsampleNearestGPUKernel() = default;
  ~UpsampleNearestGPUKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x_blob = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y_blob = ctx->Tensor4ArgNameAndIndex("y", 0);
    const float height_scale = ctx->Attr<float>("height_scale");
    const float width_scale = ctx->Attr<float>("width_scale");
    const int64_t elem_cnt = y_blob->shape().elem_cnt();
    NdIndexOffsetHelper<int64_t, 4> in_helper(x_blob->shape().At(0), x_blob->shape().At(1),
                                              x_blob->shape().At(2), x_blob->shape().At(3));
    NdIndexOffsetHelper<int64_t, 4> out_helper(y_blob->shape().At(0), y_blob->shape().At(1),
                                               y_blob->shape().At(2), y_blob->shape().At(3));

    RUN_CUDA_KERNEL((UpsampleNearestForward<T>), ctx->device_ctx(), elem_cnt, elem_cnt,
                    x_blob->dptr<T>(), in_helper, out_helper, x_blob->shape().At(2),
                    x_blob->shape().At(3), 1.f / height_scale, 1.f / width_scale,
                    y_blob->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
class UpsampleNearestGradGPUKernel final : public user_op::OpKernel {
 public:
  UpsampleNearestGradGPUKernel() = default;
  ~UpsampleNearestGradGPUKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* dx_blob = ctx->Tensor4ArgNameAndIndex("dx", 0);
    if (dx_blob == nullptr) { return; }
    Memset<DeviceType::kGPU>(ctx->device_ctx(), dx_blob->mut_dptr<T>(), 0,
                             dx_blob->shape().elem_cnt() * sizeof(T));
    const user_op::Tensor* dy_blob = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const float height_scale = ctx->Attr<float>("height_scale");
    const float width_scale = ctx->Attr<float>("width_scale");
    const int64_t elem_cnt = dy_blob->shape().elem_cnt();
    NdIndexOffsetHelper<int64_t, 4> dy_helper(dy_blob->shape().At(0), dy_blob->shape().At(1),
                                              dy_blob->shape().At(2), dy_blob->shape().At(3));
    NdIndexOffsetHelper<int64_t, 4> dx_helper(dx_blob->shape().At(0), dx_blob->shape().At(1),
                                              dx_blob->shape().At(2), dx_blob->shape().At(3));
    RUN_CUDA_KERNEL((UpsampleNearestBackward<T>), ctx->device_ctx(), elem_cnt, elem_cnt,
                    dy_blob->dptr<T>(), dy_helper, dx_helper, dx_blob->shape().At(2),
                    dx_blob->shape().At(3), 1.f / height_scale, 1.f / width_scale,
                    dx_blob->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_UPSAMPLE_NEAREST_GPU_KERNEL(dtype)                                      \
  REGISTER_USER_KERNEL("upsample")                                                       \
      .SetCreateFn<UpsampleNearestGPUKernel<dtype>>()                                    \
      .SetIsMatchedHob(                                                                  \
          (user_op::HobDeviceTag() == "gpu")                                             \
          & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value)                  \
          & (user_op::HobAttr<std::string>("interpolation") == std::string("nearest"))); \
  REGISTER_USER_KERNEL("upsample_grad")                                                  \
      .SetCreateFn<UpsampleNearestGradGPUKernel<dtype>>()                                \
      .SetIsMatchedHob(                                                                  \
          (user_op::HobDeviceTag() == "gpu")                                             \
          & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value)                 \
          & (user_op::HobAttr<std::string>("interpolation") == std::string("nearest")));

REGISTER_UPSAMPLE_NEAREST_GPU_KERNEL(float)
REGISTER_UPSAMPLE_NEAREST_GPU_KERNEL(double)

template<typename T>
class UpsampleBilinearGPUKernel final : public user_op::OpKernel {
 public:
  UpsampleBilinearGPUKernel() = default;
  ~UpsampleBilinearGPUKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x_blob = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y_blob = ctx->Tensor4ArgNameAndIndex("y", 0);
    const float height_scale = ctx->Attr<float>("height_scale");
    const float width_scale = ctx->Attr<float>("width_scale");
    const int64_t elem_cnt = y_blob->shape().elem_cnt();
    NdIndexOffsetHelper<int64_t, 4> in_helper(x_blob->shape().At(0), x_blob->shape().At(1),
                                              x_blob->shape().At(2), x_blob->shape().At(3));
    NdIndexOffsetHelper<int64_t, 4> out_helper(y_blob->shape().At(0), y_blob->shape().At(1),
                                               y_blob->shape().At(2), y_blob->shape().At(3));

    RUN_CUDA_KERNEL((UpsampleBilinearForward<T>), ctx->device_ctx(), elem_cnt, elem_cnt,
                    x_blob->dptr<T>(), in_helper, out_helper, x_blob->shape().At(2),
                    x_blob->shape().At(3), 1.f / height_scale, 1.f / width_scale,
                    y_blob->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
class UpsampleBilinearGradGPUKernel final : public user_op::OpKernel {
 public:
  UpsampleBilinearGradGPUKernel() = default;
  ~UpsampleBilinearGradGPUKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* dx_blob = ctx->Tensor4ArgNameAndIndex("dx", 0);
    if (dx_blob == nullptr) { return; }
    Memset<DeviceType::kGPU>(ctx->device_ctx(), dx_blob->mut_dptr<T>(), 0,
                             dx_blob->shape().elem_cnt() * sizeof(T));
    const user_op::Tensor* dy_blob = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const float height_scale = ctx->Attr<float>("height_scale");
    const float width_scale = ctx->Attr<float>("width_scale");
    const int64_t elem_cnt = dy_blob->shape().elem_cnt();
    NdIndexOffsetHelper<int64_t, 4> dy_helper(dy_blob->shape().At(0), dy_blob->shape().At(1),
                                              dy_blob->shape().At(2), dy_blob->shape().At(3));
    NdIndexOffsetHelper<int64_t, 4> dx_helper(dx_blob->shape().At(0), dx_blob->shape().At(1),
                                              dx_blob->shape().At(2), dx_blob->shape().At(3));

    RUN_CUDA_KERNEL((UpsampleBilinearBackward<T>), ctx->device_ctx(), elem_cnt, elem_cnt,
                    dy_blob->dptr<T>(), dy_helper, dx_helper, dx_blob->shape().At(2),
                    dx_blob->shape().At(3), 1.f / height_scale, 1.f / width_scale,
                    dx_blob->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_UPSAMPLE_BILINEAR_GPU_KERNEL(dtype)                                      \
  REGISTER_USER_KERNEL("upsample")                                                        \
      .SetCreateFn<UpsampleBilinearGPUKernel<dtype>>()                                    \
      .SetIsMatchedHob(                                                                   \
          (user_op::HobDeviceTag() == "gpu")                                              \
          & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value)                   \
          & (user_op::HobAttr<std::string>("interpolation") == std::string("bilinear"))); \
  REGISTER_USER_KERNEL("upsample_grad")                                                   \
      .SetCreateFn<UpsampleBilinearGradGPUKernel<dtype>>()                                \
      .SetIsMatchedHob(                                                                   \
          (user_op::HobDeviceTag() == "gpu")                                              \
          & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value)                  \
          & (user_op::HobAttr<std::string>("interpolation") == std::string("bilinear")));

REGISTER_UPSAMPLE_BILINEAR_GPU_KERNEL(float)
REGISTER_UPSAMPLE_BILINEAR_GPU_KERNEL(double)

}  // namespace oneflow
