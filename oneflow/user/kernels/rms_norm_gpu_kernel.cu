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
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/ep/include/primitive/fill.h"
#include "oneflow/core/ep/include/primitive/matmul.h"
#include "oneflow/core/device/cudnn_util.h"
#include "oneflow/core/cuda/atomic.cuh"
#include "oneflow/core/cuda/rms_norm.cuh"
#include <cub/cub.cuh>
#if CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#endif  // CUDA_VERSION >= 11000

namespace oneflow {
namespace cuda {
namespace rms_norm {

template<typename SRC, typename DST, bool affine>
struct AffineStore {
  AffineStore(DST* dst, const DST* weight, int32_t row_size)
      : dst(dst), weight(weight), row_size(row_size) {}

  template<int N>
  __device__ void store(const SRC* src, int32_t row, int32_t col) {
    layer_norm::Pack<DST, N> dst_pack;
    layer_norm::Pack<DST, N> weight_pack;
    const int32_t offset = (row * row_size + col) / N;
    const int32_t weight_offset = col / N;
    if (affine) {
      weight_pack.storage =
          *(reinterpret_cast<const layer_norm::PackType<DST, N>*>(weight) + weight_offset);
    }
#pragma unroll
    for (int i = 0; i < N; ++i) {
      if (affine) {
        dst_pack.elem[i] = static_cast<DST>(src[i]) * weight_pack.elem[i];
      } else {
        dst_pack.elem[i] = static_cast<DST>(src[i]);
      }
    }
    *(reinterpret_cast<layer_norm::PackType<DST, N>*>(dst) + offset) = dst_pack.storage;
  }

  DST* dst;
  const DST* weight;
  int32_t row_size;
};

template<typename SRC, typename DST, bool affine>
struct AffineLoad {
  AffineLoad(const SRC* src, const SRC* weight, int32_t row_size)
      : src(src), weight(weight), row_size(row_size) {}

  template<int N>
  __device__ void load(DST* dst, int32_t row, int32_t col) const {
    layer_norm::Pack<SRC, N> src_pack;
    layer_norm::Pack<SRC, N> weight_pack;
    const int32_t offset = (row * row_size + col) / N;
    src_pack.storage = *(reinterpret_cast<const layer_norm::PackType<SRC, N>*>(src) + offset);
    if (affine) {
      const int32_t weight_offset = col / N;
      weight_pack.storage =
          *(reinterpret_cast<const layer_norm::PackType<SRC, N>*>(weight) + weight_offset);
    }
#pragma unroll
    for (int i = 0; i < N; ++i) {
      if (affine) {
        dst[i] = static_cast<DST>(src_pack.elem[i] * weight_pack.elem[i]);
      } else {
        dst[i] = static_cast<DST>(src_pack.elem[i]);
      }
    }
  }
  const SRC* src;
  const SRC* weight;
  int32_t row_size;
};

template<typename T, bool affine>
void RmsNormForwardGpu(ep::Stream* stream, const int64_t nrow, const int64_t ncol, const double eps,
                       const T* x_dptr, const T* w_dptr, T* y_dptr, user_op::Tensor* inv_rms) {
  using ComputeType = typename layer_norm::DefaultComputeType<T>::type;
  layer_norm::DirectLoad<T, ComputeType> load(x_dptr, ncol);
  AffineStore<ComputeType, T, affine> store(y_dptr, w_dptr, ncol);
  DispatchRmsNorm<decltype(load), decltype(store), ComputeType>(
      stream->As<ep::CudaStream>()->cuda_stream(), load, store, nrow, ncol, eps,
      inv_rms->mut_dptr<ComputeType>());
}

template<typename T>
void DispatchRmsNormForwardGpu(ep::Stream* stream, const int64_t nrow, const int64_t ncol,
                               const double eps, const T* x_dptr, const T* w_dptr, T* y_dptr,
                               user_op::Tensor* inv_rms) {
  if (w_dptr) {
    RmsNormForwardGpu<T, true>(stream, nrow, ncol, eps, x_dptr, w_dptr, y_dptr, inv_rms);
  } else {
    RmsNormForwardGpu<T, false>(stream, nrow, ncol, eps, x_dptr, w_dptr, y_dptr, inv_rms);
  }
}

template<typename T, bool affine>
void RmsNormBackwardGpu(ep::Stream* stream, const int64_t nrow, const int64_t ncol,
                        const T* dy_dptr, const T* x_dptr, const user_op::Tensor* inv_rms,
                        const T* weight_dptr, T* dx_ptr) {
  using ComputeType = typename cuda::layer_norm::DefaultComputeType<T>::type;
  layer_norm::DirectLoad<T, ComputeType> load_x(x_dptr, ncol);
  AffineLoad<T, ComputeType, affine> load_dy(dy_dptr, weight_dptr, ncol);
  layer_norm::DirectStore<ComputeType, T> store(dx_ptr, ncol);
  OF_CUDA_CHECK((rms_norm::DispatchRmsNormGrad<decltype(load_x), decltype(load_dy), decltype(store),
                                               ComputeType>(
      stream->As<ep::CudaStream>()->cuda_stream(), nrow, ncol, load_x, load_dy, store,
      inv_rms->dptr<ComputeType>())));
}

template<typename T>
void DispatchRmsNormBackwardGpu(ep::Stream* stream, const int64_t nrow, const int64_t ncol,
                                const T* dy_dptr, const T* x_dptr, const user_op::Tensor* inv_rms,
                                const T* weight_dptr, T* dx_dptr) {
  if (weight_dptr) {
    RmsNormBackwardGpu<T, true>(stream, nrow, ncol, dy_dptr, x_dptr, inv_rms, weight_dptr, dx_dptr);
  } else {
    RmsNormBackwardGpu<T, false>(stream, nrow, ncol, dy_dptr, x_dptr, inv_rms, weight_dptr,
                                 dx_dptr);
  }
}

}  // namespace rms_norm

template<typename T>
class RmsNormGpuKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  RmsNormGpuKernel() = default;
  ~RmsNormGpuKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* inv_rms = ctx->Tensor4ArgNameAndIndex("inv_rms", 0);
    const double eps = ctx->Attr<float>("epsilon");
    const Shape& normalized_shape = ctx->Attr<Shape>("normalized_shape");
    const int64_t ncol = normalized_shape.elem_cnt();
    const int64_t nrow = inv_rms->shape_view().elem_cnt();
    CHECK_EQ(x->shape_view().elem_cnt(), ncol * nrow);

    const T* weight_dptr = nullptr;
    if (ctx->has_input("weight", 0)) {
      const auto* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
      CHECK_EQ(weight->shape_view().elem_cnt(), ncol);
      weight_dptr = weight->dptr<T>();
    }
    rms_norm::DispatchRmsNormForwardGpu<T>(ctx->stream(), nrow, ncol, eps, x->dptr<T>(),
                                           weight_dptr, y->mut_dptr<T>(), inv_rms);
  };
};

#define REGISTER_RMS_NORM_CUDA_KERNEL(dtype)                           \
  REGISTER_USER_KERNEL("rms_norm")                                     \
      .SetCreateFn<RmsNormGpuKernel<dtype>>()                          \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_RMS_NORM_CUDA_KERNEL(float)
REGISTER_RMS_NORM_CUDA_KERNEL(double)
REGISTER_RMS_NORM_CUDA_KERNEL(half)
#if CUDA_VERSION >= 11000
REGISTER_RMS_NORM_CUDA_KERNEL(nv_bfloat16)
#endif

template<typename T>
class RmsNormGradGpuKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  RmsNormGradGpuKernel() = default;
  ~RmsNormGradGpuKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* inv_rms = ctx->Tensor4ArgNameAndIndex("inv_rms", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const int64_t nrow = inv_rms->shape_view().elem_cnt();
    const int64_t ncol = x->shape_view().elem_cnt() / nrow;
    const T* weight_dptr = nullptr;
    if (ctx->has_input("weight", 0)) {
      const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
      CHECK_EQ(ncol, weight->shape_view().elem_cnt());
      weight_dptr = weight->dptr<T>();
    }
    rms_norm::DispatchRmsNormBackwardGpu<T>(ctx->stream(), nrow, ncol, dy->dptr<T>(), x->dptr<T>(),
                                            inv_rms, weight_dptr, dx->mut_dptr<T>());
  };
};

#define REGISTER_RMS_NORM_GRAD_CUDA_KERNEL(dtype)                      \
  REGISTER_USER_KERNEL("rms_norm_grad")                                \
      .SetCreateFn<RmsNormGradGpuKernel<dtype>>()                      \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value));

REGISTER_RMS_NORM_GRAD_CUDA_KERNEL(float)
REGISTER_RMS_NORM_GRAD_CUDA_KERNEL(double)
REGISTER_RMS_NORM_GRAD_CUDA_KERNEL(half)
#if CUDA_VERSION >= 11000
REGISTER_RMS_NORM_GRAD_CUDA_KERNEL(nv_bfloat16)
#endif

namespace {

constexpr int kNProcPerThread = 4;

}  // namespace

template<typename T>
class RmsNormParamGradGpuKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  RmsNormParamGradGpuKernel() = default;
  ~RmsNormParamGradGpuKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* inv_rms = ctx->Tensor4ArgNameAndIndex("inv_rms", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    user_op::Tensor* weight_grad = ctx->Tensor4ArgNameAndIndex("weight_grad", 0);
    const int64_t nrow = inv_rms->shape_view().elem_cnt();
    const int64_t ncol = weight_grad->shape_view().elem_cnt();

    // step 1: dx = dy * y and reduce partial rows
    const int block_dim_x = rms_norm::kWarpSize;
    const int block_dim_y = rms_norm::kWarpSize / kNProcPerThread;
    int grid_dim_x;
    int grid_dim_y;
    OF_CUDA_CHECK((rms_norm::GetGrid2Dim<kNProcPerThread, T>(nrow, ncol, block_dim_x, block_dim_y,
                                                             &grid_dim_x, &grid_dim_y)));
    // tmp weight shape [grid_dim_y, ncol] (reduce nrow -> grid_dim_y)
    size_t tmp_weight_grad_size = grid_dim_y * ncol;
    T* tmp_weight_grad_dptr = reinterpret_cast<T*>(tmp_buffer->mut_dptr());
    using ComputeType = typename layer_norm::DefaultComputeType<T>::type;
    dim3 grid_dims(grid_dim_x, grid_dim_y);
    dim3 block_dims(block_dim_x, block_dim_y);
    rms_norm::RmsNormParamGrad<kNProcPerThread, T, ComputeType>
        <<<grid_dims, block_dims, 0, ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
            nrow, ncol, dy->dptr<T>(), x->dptr<T>(), inv_rms->dptr<ComputeType>(),
            tmp_weight_grad_dptr);

    // step 2: reduce rows throught gemm to calculate weight grad
    // fill ones matrix with shape (grid_dim_y, 1)
    const int32_t m = ncol;
    const int32_t n = 1;
    const int32_t k = grid_dim_y;
    const DataType data_type = dy->data_type();
    auto fill = ep::primitive::NewPrimitive<ep::primitive::FillFactory>(
        ctx->stream()->device_type(), data_type);
    CHECK(fill);
    T* tmp_ones_dptr = tmp_buffer->mut_dptr<T>() + tmp_weight_grad_size;
    fill->Launch(ctx->stream(), tmp_ones_dptr, 1.0, k);
    // tmp weight grad (grid_dim_y, ncol) (T) * tmp ones (grid_dim_y, 1) (N) -> weight grad (ncol,
    // 1)
    auto matmul = ep::primitive::NewPrimitive<ep::primitive::MatmulFactory>(
        ctx->stream()->device_type(), data_type, ep::primitive::BlasTransposeType::T,
        ep::primitive::BlasTransposeType::N);
    CHECK(matmul);
    matmul->Launch(ctx->stream(), m, n, k, /*alpha*/ 1.0, tmp_weight_grad_dptr, tmp_ones_dptr,
                   /*beta*/ 0.0, weight_grad->mut_dptr());
  };
};

#define REGISTER_RMS_NORM_PARAM_GRAD_GPU_KERNEL(dtype)                                  \
  REGISTER_USER_KERNEL("rms_norm_param_grad")                                           \
      .SetCreateFn<RmsNormParamGradGpuKernel<dtype>>()                                  \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                  \
                       && (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                               \
        const auto& shape = ctx->InputTensorDesc("dy", 0).shape();                      \
        const auto& b_shape = ctx->InputTensorDesc("inv_rms", 0).shape();               \
        const int64_t nrow = b_shape.elem_cnt();                                        \
        const int64_t ncol = shape.elem_cnt() / nrow;                                   \
        const int block_dim_x = rms_norm::kWarpSize;                                    \
        const int block_dim_y = rms_norm::kWarpSize / kNProcPerThread;                  \
        int grid_dim_x;                                                                 \
        int grid_dim_y;                                                                 \
        OF_CUDA_CHECK((rms_norm::GetGrid2Dim<kNProcPerThread, dtype>(                   \
            nrow, ncol, block_dim_x, block_dim_y, &grid_dim_x, &grid_dim_y)));          \
        return (grid_dim_y * ncol + grid_dim_y) * sizeof(dtype);                        \
      });

REGISTER_RMS_NORM_PARAM_GRAD_GPU_KERNEL(float)
REGISTER_RMS_NORM_PARAM_GRAD_GPU_KERNEL(double)
REGISTER_RMS_NORM_PARAM_GRAD_GPU_KERNEL(half)
#if CUDA_VERSION >= 11000
REGISTER_RMS_NORM_PARAM_GRAD_GPU_KERNEL(nv_bfloat16)
#endif

}  // namespace cuda
}  // namespace oneflow
