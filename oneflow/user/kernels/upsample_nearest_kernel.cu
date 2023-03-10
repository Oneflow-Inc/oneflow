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
#include "oneflow/core/kernel/kernel_util.cuh"
#include "oneflow/user/kernels/upsample_kernel.h"
#include "oneflow/core/kernel/cuda_graph_support.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void UpsampleNearest1DForward(const int64_t elem_cnt, const T* in_dptr,
                                         NdIndexOffsetHelper<int64_t, 3> in_helper,
                                         NdIndexOffsetHelper<int64_t, 3> out_helper,
                                         const int64_t in_height, const double scale_factor,
                                         T* out_dptr) {
  CUDA_1D_KERNEL_LOOP(index, elem_cnt) {
    int64_t n, c, h;
    out_helper.OffsetToNdIndex(index, n, c, h);
    const int64_t in_h = GetNearestInputIndex(h, scale_factor, in_height);
    out_dptr[index] = in_dptr[in_helper.NdIndexToOffset(n, c, in_h)];
  }
}

template<typename T>
__global__ void UpsampleNearest1DBackward(const int64_t elem_cnt, const T* dy_dptr,
                                          NdIndexOffsetHelper<int64_t, 3> dy_helper,
                                          NdIndexOffsetHelper<int64_t, 3> dx_helper,
                                          const int64_t in_height, const double scale_factor,
                                          T* dx_dptr) {
  CUDA_1D_KERNEL_LOOP(index, elem_cnt) {
    int64_t n, c, h;
    dy_helper.OffsetToNdIndex(index, n, c, h);
    const int64_t dx_h = GetNearestInputIndex(h, scale_factor, in_height);
    cuda::atomic::FastAdd(dx_dptr, dx_helper.NdIndexToOffset(n, c, dx_h), elem_cnt,
                          static_cast<T>(dy_dptr[index]));
  }
}

template<typename T>
__global__ void UpsampleNearest2DForward(const int64_t elem_cnt, const T* in_dptr,
                                         NdIndexOffsetHelper<int64_t, 4> in_helper,
                                         NdIndexOffsetHelper<int64_t, 4> out_helper,
                                         const int64_t in_height, const int64_t in_width,
                                         const double scale_h, const double scale_w, T* out_dptr) {
  CUDA_1D_KERNEL_LOOP(index, elem_cnt) {
    int64_t n, c, h, w;
    out_helper.OffsetToNdIndex(index, n, c, h, w);
    const int64_t in_h = GetNearestInputIndex(h, scale_h, in_height);
    const int64_t in_w = GetNearestInputIndex(w, scale_w, in_width);
    out_dptr[index] = in_dptr[in_helper.NdIndexToOffset(n, c, in_h, in_w)];
  }
}

template<typename T>
struct alignas(2 * sizeof(T)) Pack2X {
  T x;
  T y;
};

template<typename T>
__global__ void UpsampleNearest2D2XForward(const int32_t in_elem_cnt, const T* in_dptr,
                                           const int32_t in_height, const int32_t in_width,
                                           T* out_dptr) {
  const int32_t in_hw_size = in_width * in_height;
  CUDA_1D_KERNEL_LOOP(index, in_elem_cnt) {
    const T in_value = in_dptr[index];
    const int32_t nc_idx = index / in_hw_size;
    const int32_t hw_off = index - nc_idx * in_hw_size;
    const int32_t h = hw_off / in_width;
    const int32_t w = hw_off - h * in_width;
    Pack2X<T> out_value{in_value, in_value};
    Pack2X<T>* out_pack_dptr = reinterpret_cast<Pack2X<T>*>(out_dptr);
    out_pack_dptr[nc_idx * in_hw_size * 2 + h * 2 * in_width + w] = out_value;
    out_pack_dptr[nc_idx * in_hw_size * 2 + (h * 2 + 1) * in_width + w] = out_value;
  }
}

template<typename T>
__global__ void UpsampleNearest2DBackward(const int64_t elem_cnt, const T* dy_dptr,
                                          NdIndexOffsetHelper<int64_t, 4> dy_helper,
                                          NdIndexOffsetHelper<int64_t, 4> dx_helper,
                                          const int64_t dx_height, const int64_t dx_width,
                                          const double scale_h, const double scale_w, T* dx_dptr) {
  CUDA_1D_KERNEL_LOOP(index, elem_cnt) {
    int64_t n, c, h, w;
    dy_helper.OffsetToNdIndex(index, n, c, h, w);
    const int64_t dx_h = GetNearestInputIndex(h, scale_h, dx_height);
    const int64_t dx_w = GetNearestInputIndex(w, scale_w, dx_width);
    cuda::atomic::FastAdd(dx_dptr, dx_helper.NdIndexToOffset(n, c, dx_h, dx_w), elem_cnt,
                          static_cast<T>(dy_dptr[index]));
  }
}

template<typename T>
__global__ void UpsampleNearest2D2XBackward(const int32_t in_elem_cnt, const T* dy_dptr,
                                            const int32_t dx_height, const int32_t dx_width,
                                            T* dx_dptr) {
  const int32_t dx_hw_size = dx_height * dx_width;
  CUDA_1D_KERNEL_LOOP(index, in_elem_cnt) {
    T dx_value = 0.0;
    const int32_t nc_idx = index / dx_hw_size;
    const int32_t dx_hw_off = index - nc_idx * dx_hw_size;
    const int32_t dx_h = dx_hw_off / dx_width;
    const int32_t dx_w = dx_hw_off - dx_h * dx_width;
    const Pack2X<T>* dy_pack_dptr = reinterpret_cast<const Pack2X<T>*>(dy_dptr);
    const Pack2X<T> dy_pack_value1 =
        dy_pack_dptr[nc_idx * dx_hw_size * 2 + dx_h * 2 * dx_width + dx_w];
    const Pack2X<T> dy_pack_value2 =
        dy_pack_dptr[nc_idx * dx_hw_size * 2 + (dx_h * 2 + 1) * dx_width + dx_w];
    dx_value += dy_pack_value1.x;
    dx_value += dy_pack_value1.y;
    dx_value += dy_pack_value2.x;
    dx_value += dy_pack_value2.y;
    dx_dptr[index] = dx_value;
  }
}

template<typename T>
__global__ void UpsampleNearest3DForward(const int64_t elem_cnt, const T* in_dptr,
                                         NdIndexOffsetHelper<int64_t, 5> in_helper,
                                         NdIndexOffsetHelper<int64_t, 5> out_helper,
                                         const int64_t in_depth, const int64_t in_height,
                                         const int64_t in_width, const float scale_d,
                                         const float scale_h, const float scale_w, T* out_dptr) {
  CUDA_1D_KERNEL_LOOP(index, elem_cnt) {
    int64_t n, c, d, h, w;
    out_helper.OffsetToNdIndex(index, n, c, d, h, w);
    const int64_t in_h = GetNearestInputIndex(h, scale_h, in_height);
    const int64_t in_w = GetNearestInputIndex(w, scale_w, in_width);
    const int64_t in_d = GetNearestInputIndex(d, scale_d, in_depth);
    out_dptr[index] = in_dptr[in_helper.NdIndexToOffset(n, c, in_d, in_h, in_w)];
  }
}

template<typename T>
__global__ void UpsampleNearest3DBackward(const int64_t elem_cnt, const T* dy_dptr,
                                          NdIndexOffsetHelper<int64_t, 5> dy_helper,
                                          NdIndexOffsetHelper<int64_t, 5> dx_helper,
                                          const int64_t in_depth, const int64_t in_height,
                                          const int64_t in_width, const float scale_d,
                                          const float scale_h, const float scale_w, T* dx_dptr) {
  CUDA_1D_KERNEL_LOOP(index, elem_cnt) {
    int64_t n, c, d, h, w;
    dy_helper.OffsetToNdIndex(index, n, c, d, h, w);
    const int64_t dx_h = GetNearestInputIndex(h, scale_h, in_height);
    const int64_t dx_w = GetNearestInputIndex(w, scale_w, in_width);
    const int64_t in_d = GetNearestInputIndex(d, scale_d, in_depth);
    cuda::atomic::FastAdd(dx_dptr, dx_helper.NdIndexToOffset(n, c, in_d, dx_h, dx_w), elem_cnt,
                          static_cast<T>(dy_dptr[index]));
  }
}

}  // namespace

template<typename T>
class UpsampleNearest1DGPUKernel final : public user_op::OpKernel {
 public:
  UpsampleNearest1DGPUKernel() = default;
  ~UpsampleNearest1DGPUKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y_tensor = ctx->Tensor4ArgNameAndIndex("y", 0);
    const std::vector<int64_t> output_size = ctx->Attr<std::vector<int64_t>>("output_size");
    double height_scale = ctx->Attr<double>("scale_factor");
    const int64_t elem_cnt = y_tensor->shape_view().elem_cnt();
    const int64_t in_height = x_tensor->shape_view().At(2);
    const int64_t out_height = y_tensor->shape_view().At(2);
    if (!output_size.empty()) {
      height_scale = static_cast<double>(out_height) / static_cast<double>(in_height);
    }
    if (in_height == out_height) {
      Memcpy<DeviceType::kCUDA>(
          ctx->stream(), y_tensor->mut_dptr<void>(), x_tensor->dptr<void>(),
          x_tensor->shape_view().elem_cnt() * GetSizeOfDataType(x_tensor->data_type()));
    } else {
      NdIndexOffsetHelper<int64_t, 3> in_helper(
          x_tensor->shape_view().At(0), x_tensor->shape_view().At(1), x_tensor->shape_view().At(2));
      NdIndexOffsetHelper<int64_t, 3> out_helper(
          y_tensor->shape_view().At(0), y_tensor->shape_view().At(1), y_tensor->shape_view().At(2));
      RUN_CUDA_KERNEL((UpsampleNearest1DForward<T>), ctx->stream(), elem_cnt, elem_cnt,
                      x_tensor->dptr<T>(), in_helper, out_helper, x_tensor->shape_view().At(2),
                      1.f / height_scale, y_tensor->mut_dptr<T>());
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
class UpsampleNearestGrad1DGPUKernel final : public user_op::OpKernel {
 public:
  UpsampleNearestGrad1DGPUKernel() = default;
  ~UpsampleNearestGrad1DGPUKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* dx_tensor = ctx->Tensor4ArgNameAndIndex("dx", 0);

    const user_op::Tensor* dy_tensor = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const std::vector<int64_t> output_size = ctx->Attr<std::vector<int64_t>>("output_size");
    double height_scale = ctx->Attr<double>("scale_factor");
    const int64_t elem_cnt = dy_tensor->shape_view().elem_cnt();
    const int64_t in_height = dx_tensor->shape_view().At(2);
    const int64_t out_height = dy_tensor->shape_view().At(2);
    if (!output_size.empty()) {
      height_scale = static_cast<double>(out_height) / static_cast<double>(in_height);
    }
    if (in_height == out_height) {
      Memcpy<DeviceType::kCUDA>(
          ctx->stream(), dx_tensor->mut_dptr<void>(), dy_tensor->dptr<void>(),
          dy_tensor->shape_view().elem_cnt() * GetSizeOfDataType(dy_tensor->data_type()));
    } else {
      Memset<DeviceType::kCUDA>(ctx->stream(), dx_tensor->mut_dptr<T>(), 0,
                                dx_tensor->shape_view().elem_cnt() * sizeof(T));
      NdIndexOffsetHelper<int64_t, 3> dy_helper(dy_tensor->shape_view().At(0),
                                                dy_tensor->shape_view().At(1),
                                                dy_tensor->shape_view().At(2));
      NdIndexOffsetHelper<int64_t, 3> dx_helper(dx_tensor->shape_view().At(0),
                                                dx_tensor->shape_view().At(1),
                                                dx_tensor->shape_view().At(2));
      RUN_CUDA_KERNEL((UpsampleNearest1DBackward<T>), ctx->stream(), elem_cnt, elem_cnt,
                      dy_tensor->dptr<T>(), dy_helper, dx_helper, dx_tensor->shape_view().At(2),
                      1.f / height_scale, dx_tensor->mut_dptr<T>());
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_UPSAMPNEAREST1D_CUDA_KERNEL(dtype)                                     \
  REGISTER_USER_KERNEL("upsample_nearest_1d")                                           \
      .SetCreateFn<UpsampleNearest1DGPUKernel<dtype>>()                                 \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                  \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("upsample_nearest_1d_grad")                                      \
      .SetCreateFn<UpsampleNearestGrad1DGPUKernel<dtype>>()                             \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                  \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_UPSAMPNEAREST1D_CUDA_KERNEL(float)
REGISTER_UPSAMPNEAREST1D_CUDA_KERNEL(double)

template<typename T>
class UpsampleNearest2DGPUKernel final : public user_op::OpKernel,
                                         public user_op::CudaGraphSupport {
 public:
  UpsampleNearest2DGPUKernel() = default;
  ~UpsampleNearest2DGPUKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y_tensor = ctx->Tensor4ArgNameAndIndex("y", 0);
    const std::vector<int64_t> output_size = ctx->Attr<std::vector<int64_t>>("output_size");
    double height_scale = ctx->Attr<double>("height_scale");
    double width_scale = ctx->Attr<double>("width_scale");
    const int64_t out_elem_cnt = y_tensor->shape_view().elem_cnt();
    const int64_t in_elem_cnt = x_tensor->shape_view().elem_cnt();
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
      const int64_t n = x_tensor->shape_view().At(0);
      const int64_t c = x_tensor->shape_view().At(1);
      if (out_height == 2 * in_height && out_width == 2 * in_width && in_elem_cnt <= 1 << 29) {
        RUN_CUDA_KERNEL(UpsampleNearest2D2XForward<T>, ctx->stream(), in_elem_cnt, in_elem_cnt,
                        x_tensor->dptr<T>(), in_height, in_width, y_tensor->mut_dptr<T>());
      } else {
        NdIndexOffsetHelper<int64_t, 4> in_helper(n, c, in_height, in_width);
        NdIndexOffsetHelper<int64_t, 4> out_helper(n, c, out_height, out_width);
        RUN_CUDA_KERNEL((UpsampleNearest2DForward<T>), ctx->stream(), out_elem_cnt, out_elem_cnt,
                        x_tensor->dptr<T>(), in_helper, out_helper, in_height, in_width,
                        1.f / height_scale, 1.f / width_scale, y_tensor->mut_dptr<T>());
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
class UpsampleNearest2DGradGPUKernel final : public user_op::OpKernel {
 public:
  UpsampleNearest2DGradGPUKernel() = default;
  ~UpsampleNearest2DGradGPUKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* dx_tensor = ctx->Tensor4ArgNameAndIndex("dx", 0);

    const user_op::Tensor* dy_tensor = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const std::vector<int64_t> output_size = ctx->Attr<std::vector<int64_t>>("output_size");
    double height_scale = ctx->Attr<double>("height_scale");
    double width_scale = ctx->Attr<double>("width_scale");
    const int64_t elem_cnt = dy_tensor->shape_view().elem_cnt();
    const int64_t in_elem_cnt = dx_tensor->shape_view().elem_cnt();
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
      if (out_height == 2 * in_height && out_width == 2 * in_width && in_elem_cnt <= 1 << 29) {
        RUN_CUDA_KERNEL(UpsampleNearest2D2XBackward<T>, ctx->stream(), in_elem_cnt, in_elem_cnt,
                        dy_tensor->dptr<T>(), dx_tensor->shape_view().At(2),
                        dx_tensor->shape_view().At(3), dx_tensor->mut_dptr<T>());
      } else {
        Memset<DeviceType::kCUDA>(ctx->stream(), dx_tensor->mut_dptr<T>(), 0,
                                  dx_tensor->shape_view().elem_cnt() * sizeof(T));
        NdIndexOffsetHelper<int64_t, 4> dy_helper(
            dy_tensor->shape_view().At(0), dy_tensor->shape_view().At(1),
            dy_tensor->shape_view().At(2), dy_tensor->shape_view().At(3));
        NdIndexOffsetHelper<int64_t, 4> dx_helper(
            dx_tensor->shape_view().At(0), dx_tensor->shape_view().At(1),
            dx_tensor->shape_view().At(2), dx_tensor->shape_view().At(3));
        RUN_CUDA_KERNEL((UpsampleNearest2DBackward<T>), ctx->stream(), elem_cnt, elem_cnt,
                        dy_tensor->dptr<T>(), dy_helper, dx_helper, dx_tensor->shape_view().At(2),
                        dx_tensor->shape_view().At(3), 1.f / height_scale, 1.f / width_scale,
                        dx_tensor->mut_dptr<T>());
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_UPSAMPLE_NEAREST_2D_CUDA_KERNEL(dtype)                                 \
  REGISTER_USER_KERNEL("upsample_nearest_2d")                                           \
      .SetCreateFn<UpsampleNearest2DGPUKernel<dtype>>()                                 \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                  \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("upsample_nearest_2d_grad")                                      \
      .SetCreateFn<UpsampleNearest2DGradGPUKernel<dtype>>()                             \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                  \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_UPSAMPLE_NEAREST_2D_CUDA_KERNEL(float)
REGISTER_UPSAMPLE_NEAREST_2D_CUDA_KERNEL(half)
REGISTER_UPSAMPLE_NEAREST_2D_CUDA_KERNEL(double)

template<typename T>
class UpsampleNearest3DGPUKernel final : public user_op::OpKernel {
 public:
  UpsampleNearest3DGPUKernel() = default;
  ~UpsampleNearest3DGPUKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y_tensor = ctx->Tensor4ArgNameAndIndex("y", 0);
    const std::vector<int64_t> output_size = ctx->Attr<std::vector<int64_t>>("output_size");
    double depth_scale = ctx->Attr<double>("depth_scale");
    double height_scale = ctx->Attr<double>("height_scale");
    double width_scale = ctx->Attr<double>("width_scale");
    const int64_t in_depth = x_tensor->shape_view().At(2);
    const int64_t in_height = x_tensor->shape_view().At(3);
    const int64_t in_width = x_tensor->shape_view().At(4);
    const int64_t out_depth = y_tensor->shape_view().At(2);
    const int64_t out_height = y_tensor->shape_view().At(3);
    const int64_t out_width = y_tensor->shape_view().At(4);
    const int64_t elem_cnt = y_tensor->shape_view().elem_cnt();
    if (!output_size.empty()) {
      depth_scale = static_cast<double>(out_depth) / static_cast<double>(in_depth);
      height_scale = static_cast<double>(out_height) / static_cast<double>(in_height);
      width_scale = static_cast<double>(out_width) / static_cast<double>(in_width);
    }
    NdIndexOffsetHelper<int64_t, 5> in_helper(
        x_tensor->shape_view().At(0), x_tensor->shape_view().At(1), x_tensor->shape_view().At(2),
        x_tensor->shape_view().At(3), x_tensor->shape_view().At(4));
    NdIndexOffsetHelper<int64_t, 5> out_helper(
        y_tensor->shape_view().At(0), y_tensor->shape_view().At(1), y_tensor->shape_view().At(2),
        y_tensor->shape_view().At(3), y_tensor->shape_view().At(4));
    RUN_CUDA_KERNEL((UpsampleNearest3DForward<T>), ctx->stream(), elem_cnt, elem_cnt,
                    x_tensor->dptr<T>(), in_helper, out_helper, x_tensor->shape_view().At(2),
                    x_tensor->shape_view().At(3), x_tensor->shape_view().At(4), 1.f / depth_scale,
                    1.f / height_scale, 1.f / width_scale, y_tensor->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
class UpsampleNearestGrad3DGPUKernel final : public user_op::OpKernel {
 public:
  UpsampleNearestGrad3DGPUKernel() = default;
  ~UpsampleNearestGrad3DGPUKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* dx_tensor = ctx->Tensor4ArgNameAndIndex("dx", 0);

    Memset<DeviceType::kCUDA>(ctx->stream(), dx_tensor->mut_dptr<T>(), 0,
                              dx_tensor->shape_view().elem_cnt() * sizeof(T));
    const user_op::Tensor* dy_tensor = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const std::vector<int64_t> output_size = ctx->Attr<std::vector<int64_t>>("output_size");
    double depth_scale = ctx->Attr<double>("depth_scale");
    double height_scale = ctx->Attr<double>("height_scale");
    double width_scale = ctx->Attr<double>("width_scale");
    const int64_t in_depth = dx_tensor->shape_view().At(2);
    const int64_t in_height = dx_tensor->shape_view().At(3);
    const int64_t in_width = dx_tensor->shape_view().At(4);
    const int64_t out_depth = dy_tensor->shape_view().At(2);
    const int64_t out_height = dy_tensor->shape_view().At(3);
    const int64_t out_width = dy_tensor->shape_view().At(4);
    const int64_t elem_cnt = dy_tensor->shape_view().elem_cnt();
    if (!output_size.empty()) {
      depth_scale = static_cast<double>(out_depth) / static_cast<double>(in_depth);
      height_scale = static_cast<double>(out_height) / static_cast<double>(in_height);
      width_scale = static_cast<double>(out_width) / static_cast<double>(in_width);
    }
    NdIndexOffsetHelper<int64_t, 5> dy_helper(
        dy_tensor->shape_view().At(0), dy_tensor->shape_view().At(1), dy_tensor->shape_view().At(2),
        dy_tensor->shape_view().At(3), dy_tensor->shape_view().At(4));
    NdIndexOffsetHelper<int64_t, 5> dx_helper(
        dx_tensor->shape_view().At(0), dx_tensor->shape_view().At(1), dx_tensor->shape_view().At(2),
        dx_tensor->shape_view().At(3), dx_tensor->shape_view().At(4));
    RUN_CUDA_KERNEL((UpsampleNearest3DBackward<T>), ctx->stream(), elem_cnt, elem_cnt,
                    dy_tensor->dptr<T>(), dy_helper, dx_helper, dx_tensor->shape_view().At(2),
                    dx_tensor->shape_view().At(3), dx_tensor->shape_view().At(4), 1.f / depth_scale,
                    1.f / height_scale, 1.f / width_scale, dx_tensor->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_UPSAMPNEAREST3D_CUDA_KERNEL(dtype)                                     \
  REGISTER_USER_KERNEL("upsample_nearest_3d")                                           \
      .SetCreateFn<UpsampleNearest3DGPUKernel<dtype>>()                                 \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                  \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("upsample_nearest_3d_grad")                                      \
      .SetCreateFn<UpsampleNearestGrad3DGPUKernel<dtype>>()                             \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                  \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_UPSAMPNEAREST3D_CUDA_KERNEL(float)
REGISTER_UPSAMPNEAREST3D_CUDA_KERNEL(double)

}  // namespace oneflow
