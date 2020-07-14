/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void UpsampleNearestForward(const int64_t nthreads, const T* in_dptr,
                                       const int64_t channel_num, const int64_t in_height,
                                       const int64_t in_width, const int64_t out_height,
                                       const int64_t out_width, const float scale_h,
                                       const float scale_w, const bool align_corners, T* out_dptr) {
  const int64_t new_area = out_height * out_width;
  const int64_t channel_area = channel_num * in_height * in_width;
  const int64_t channel_new_area = channel_num * out_height * out_width;
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int64_t h = (index / out_width) % out_height;
    const int64_t w = index % out_width;
    const int64_t c = (index / new_area) % channel_num;
    const int64_t n = index / channel_new_area;
    const int64_t in_h = min((align_corners) ? static_cast<int64_t>(roundf(h * scale_h))
                                             : static_cast<int64_t>(floorf(h * scale_h)),
                             in_height - 1);
    const int64_t in_w = min((align_corners) ? static_cast<int64_t>(roundf(w * scale_w))
                                             : static_cast<int64_t>(floorf(w * scale_w)),
                             in_width - 1);
    out_dptr[index] = in_dptr[n * channel_area + (c * in_height + in_h) * in_width + in_w];
  }
}

template<typename T>
__global__ void UpsampleNearestBackward(const int64_t nthreads, const T* dy_dptr,
                                        const int64_t channel_num, const int64_t dx_height,
                                        const int64_t dx_width, const int64_t dy_height,
                                        const int64_t dy_width, const float scale_h,
                                        const float scale_w, const bool align_corners, T* dx_dptr) {
  const int64_t area = dx_height * dx_width;
  const int64_t new_area = dy_height * dy_width;
  const int64_t channel_area = channel_num * dx_height * dx_width;
  const int64_t channel_new_area = channel_num * dy_height * dy_width;
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int64_t h = (index / dy_width) % dy_height;
    const int64_t w = index % dy_width;
    const int64_t c = (index / new_area) % channel_num;
    const int64_t n = index / channel_new_area;
    const int64_t in_h = min((align_corners) ? static_cast<int64_t>(roundf(h * scale_h))
                                             : static_cast<int64_t>(floorf(h * scale_h)),
                             dx_height - 1);
    const int64_t in_w = min((align_corners) ? static_cast<int64_t>(roundf(w * scale_w))
                                             : static_cast<int64_t>(floorf(w * scale_w)),
                             dx_width - 1);
    atomicAdd(dx_dptr + n * channel_area + (c * dx_height + in_h) * dx_width + in_w,
              dy_dptr[index]);
  }
}

template<typename T>
__global__ void UpsampleBilinearForward(const int64_t nthreads, const T* in_dptr,
                                        const int64_t channel_num, const int64_t in_height,
                                        const int64_t in_width, const int64_t out_height,
                                        const int64_t out_width, const float scale_h,
                                        const float scale_w, T* out_dptr) {
  const int64_t new_area = out_height * out_width;
  const int64_t channel_area = channel_num * in_height * in_width;
  const int64_t channel_new_area = channel_num * out_height * out_width;
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int64_t h = (index / out_width) % out_height;
    const int64_t w = index % out_width;
    const int64_t c = (index / new_area) % channel_num;
    const int64_t n = index / channel_new_area;

    const float in_h = (static_cast<float>(h) + 0.5f) * scale_h - 0.5f;
    const int top_h_index = in_h > 0.0 ? floorf(in_h) : 0;
    const int bottom_h_index = (in_h < in_height - 1) ? ceilf(in_h) : in_height - 1;
    const float h_lerp = in_h - top_h_index;

    const float in_w = (static_cast<float>(w) + 0.5f) * scale_w - 0.5f;
    const int left_w_index = in_w > 0.0 ? floorf(in_w) : 0;
    const int right_w_index = (in_w < in_width - 1) ? ceilf(in_w) : in_width - 1;
    const float w_lerp = in_w - left_w_index;
    const float top_left(
        in_dptr[n * channel_area + (c * in_height + top_h_index) * in_width + left_w_index]);
    const float top_right(
        in_dptr[n * channel_area + (c * in_height + top_h_index) * in_width + right_w_index]);
    const float bottom_left(
        in_dptr[n * channel_area + (c * in_height + bottom_h_index) * in_width + left_w_index]);
    const float bottom_right(
        in_dptr[n * channel_area + (c * in_height + bottom_h_index) * in_width + right_w_index]);
    const float top = top_left + (top_right - top_left) * w_lerp;
    const float bottom = bottom_left + (bottom_right - bottom_left) * w_lerp;
    out_dptr[index] = top + (bottom - top) * h_lerp;
  }
}

template<typename T>
__global__ void UpsampleBilinearBackward(const int64_t nthreads, const T* dy_dptr,
                                         const int64_t channel_num, const int64_t dx_height,
                                         const int64_t dx_width, const int64_t dy_height,
                                         const int64_t dy_width, const float scale_h,
                                         const float scale_w, T* dx_dptr) {
  const int64_t area = dx_height * dx_width;
  const int64_t new_area = dy_height * dy_width;
  const int64_t channel_area = channel_num * dx_height * dx_width;
  const int64_t channel_new_area = channel_num * dy_height * dy_width;
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int64_t h = (index / dy_width) % dy_height;
    const int64_t w = index % dy_width;
    const int64_t c = (index / new_area) % channel_num;
    const int64_t n = index / channel_new_area;

    const float original_h = (static_cast<float>(h) + 0.5f) * scale_h - 0.5f;
    const int top_h_index = original_h > 0.0 ? floorf(original_h) : 0;
    const int bottom_h_index = (original_h < dx_height - 1) ? ceilf(original_h) : dx_height - 1;
    const float h_lerp = original_h - floorf(original_h);

    const float original_w = (static_cast<float>(w) + 0.5f) * scale_w - 0.5f;
    const int left_w_index = original_w > 0.0 ? floorf(original_w) : 0;
    const int right_w_index = (original_w < dx_width - 1) ? ceilf(original_w) : dx_width - 1;
    const float w_lerp = original_w - floorf(original_w);

    const float dtop = (1 - h_lerp) * dy_dptr[index];
    atomicAdd(dx_dptr + n * channel_area + (c * dx_height + top_h_index) * dx_width + left_w_index,
              static_cast<T>((1 - w_lerp) * dtop));
    atomicAdd(dx_dptr + n * channel_area + (c * dx_height + top_h_index) * dx_width + right_w_index,
              static_cast<T>(w_lerp * dtop));
    const float dbottom = h_lerp * dy_dptr[index];
    atomicAdd(
        dx_dptr + n * channel_area + (c * dx_height + bottom_h_index) * dx_width + left_w_index,
        static_cast<T>((1 - w_lerp) * dbottom));
    atomicAdd(
        dx_dptr + n * channel_area + (c * dx_height + bottom_h_index) * dx_width + right_w_index,
        static_cast<T>(w_lerp * dbottom));
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
    UpsampleNearestForward<T>
        <<<BlocksNum4ThreadsNum(elem_cnt), 1024, 0, ctx->device_ctx()->cuda_stream()>>>(
            elem_cnt, x_blob->dptr<T>(), x_blob->shape().At(1), x_blob->shape().At(2),
            x_blob->shape().At(3), y_blob->shape().At(2), y_blob->shape().At(3), 1.f / height_scale,
            1.f / width_scale, false, y_blob->mut_dptr<T>());
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
    UpsampleNearestBackward<T>
        <<<BlocksNum4ThreadsNum(elem_cnt), 1024, 0, ctx->device_ctx()->cuda_stream()>>>(
            elem_cnt, dy_blob->dptr<T>(), dx_blob->shape().At(1), dx_blob->shape().At(2),
            dx_blob->shape().At(3), dy_blob->shape().At(2), dy_blob->shape().At(3),
            1.f / height_scale, 1.f / width_scale, false, dx_blob->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_UPSAMPLE_NEAREST_GPU_KERNEL(dtype)                                      \
  REGISTER_USER_KERNEL("upsample")                                                       \
      .SetCreateFn<UpsampleNearestGPUKernel<dtype>>()                                    \
      .SetIsMatchedHob(                                                                  \
          (user_op::HobDeviceType() == DeviceType::kGPU)                                 \
          & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value)                  \
          & (user_op::HobAttr<std::string>("interpolation") == std::string("nearest"))); \
  REGISTER_USER_KERNEL("upsample_grad")                                                  \
      .SetCreateFn<UpsampleNearestGradGPUKernel<dtype>>()                                \
      .SetIsMatchedHob(                                                                  \
          (user_op::HobDeviceType() == DeviceType::kGPU)                                 \
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
    UpsampleBilinearForward<T>
        <<<BlocksNum4ThreadsNum(elem_cnt), 1024, 0, ctx->device_ctx()->cuda_stream()>>>(
            elem_cnt, x_blob->dptr<T>(), x_blob->shape().At(1), x_blob->shape().At(2),
            x_blob->shape().At(3), y_blob->shape().At(2), y_blob->shape().At(3), 1.f / height_scale,
            1.f / width_scale, y_blob->mut_dptr<T>());
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
    UpsampleBilinearBackward<T>
        <<<BlocksNum4ThreadsNum(elem_cnt), 1024, 0, ctx->device_ctx()->cuda_stream()>>>(
            elem_cnt, dy_blob->dptr<T>(), dx_blob->shape().At(1), dx_blob->shape().At(2),
            dx_blob->shape().At(3), dy_blob->shape().At(2), dy_blob->shape().At(3),
            1.f / height_scale, 1.f / width_scale, dx_blob->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_UPSAMPLE_BILINEAR_GPU_KERNEL(dtype)                                      \
  REGISTER_USER_KERNEL("upsample")                                                        \
      .SetCreateFn<UpsampleBilinearGPUKernel<dtype>>()                                    \
      .SetIsMatchedHob(                                                                   \
          (user_op::HobDeviceType() == DeviceType::kGPU)                                  \
          & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value)                   \
          & (user_op::HobAttr<std::string>("interpolation") == std::string("bilinear"))); \
  REGISTER_USER_KERNEL("upsample_grad")                                                   \
      .SetCreateFn<UpsampleBilinearGradGPUKernel<dtype>>()                                \
      .SetIsMatchedHob(                                                                   \
          (user_op::HobDeviceType() == DeviceType::kGPU)                                  \
          & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value)                  \
          & (user_op::HobAttr<std::string>("interpolation") == std::string("bilinear")));

REGISTER_UPSAMPLE_BILINEAR_GPU_KERNEL(float)
REGISTER_UPSAMPLE_BILINEAR_GPU_KERNEL(double)

}  // namespace oneflow
