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
void RmsNormForwardGpu(ep::Stream* stream, const int64_t nrows, const int64_t ncols,
                       const double eps, const T* x_dptr, const T* w_dptr, T* y_dptr,
                       user_op::Tensor* inv_rms) {
  using ComputeType = typename layer_norm::DefaultComputeType<T>::type;
  layer_norm::DirectLoad<T, ComputeType> load(x_dptr, ncols);
  AffineStore<ComputeType, T, affine> store(y_dptr, w_dptr, ncols);
  DispatchRmsNorm<decltype(load), decltype(store), ComputeType>(
      stream->As<ep::CudaStream>()->cuda_stream(), load, store, nrows, ncols, eps,
      inv_rms->mut_dptr<ComputeType>());
}

template<typename T>
void DispatchRmsNormForwardGpu(ep::Stream* stream, const int64_t nrows, const int64_t ncols,
                               const double eps, const T* x_dptr, const T* w_dptr, T* y_dptr,
                               user_op::Tensor* inv_rms) {
  if (w_dptr) {
    RmsNormForwardGpu<T, true>(stream, nrows, ncols, eps, x_dptr, w_dptr, y_dptr, inv_rms);
  } else {
    RmsNormForwardGpu<T, false>(stream, nrows, ncols, eps, x_dptr, w_dptr, y_dptr, inv_rms);
  }
}

constexpr int tile_size = 32;
constexpr int num_per_block = 4;
constexpr int block_dim_x = 32;
constexpr int block_dim_y = 32 / num_per_block;

template<typename T, typename ComputeType>
__global__ void RmsNormParamGrad(int nrows, int ncols, const T* __restrict__ dy,
                                 const T* __restrict__ x, const ComputeType* __restrict__ inv_rms,
                                 T* __restrict__ b_weight_grad) {
  __shared__ ComputeType dweight[32][33];
  ComputeType dweight_sum[num_per_block];
#pragma unroll
  for (int index = 0; index < num_per_block; ++index) { dweight_sum[index] = 0; }
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < ncols) {
    for (int i = blockIdx.y * tile_size + threadIdx.y; i < nrows; i += tile_size * gridDim.y) {
#pragma unroll
      for (int index = 0; index < num_per_block; ++index) {
        int row = i + index * blockDim.y;
        if (row < nrows) {
          int offset = row * ncols + col;
          const ComputeType dy_val = static_cast<ComputeType>(dy[offset]);
          const ComputeType x_val = static_cast<ComputeType>(x[offset]);
          const ComputeType inv_rms_val = inv_rms[row];
          dweight_sum[index] += dy_val * x_val * inv_rms_val;
        }
      }
    }
  }
#pragma unroll
  for (int index = 0; index < num_per_block; ++index) {
    dweight[index * blockDim.y + threadIdx.y][threadIdx.x] = dweight_sum[index];
  }
  __syncthreads();
#pragma unroll
  for (int index = 0; index < num_per_block; ++index) {
    const int col = blockIdx.x * blockDim.x + threadIdx.y + index * blockDim.y;
    if (col < ncols) {
      ComputeType dweight_val = dweight[threadIdx.x][threadIdx.y + index * blockDim.y];
      ComputeType global_dweight = WarpReduceSum<ComputeType>(dweight_val);
      if (threadIdx.x == 0) {
        const int offset = blockIdx.y * ncols + col;
        b_weight_grad[offset] = global_dweight;
      }
    }
  }
}

template<typename T>
int GetGirdDimY(const int64_t num_instances, const int64_t norm_size) {
  using ComputeType = typename layer_norm::DefaultComputeType<T>::type;
  const int grid_dim_x = (norm_size + tile_size - 1) / tile_size;
  const int max_grid_dim_y = (num_instances + tile_size - 1) / tile_size;
  const int block_size = block_dim_x * block_dim_y;
  int max_active_blocks = 0;
  OF_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_blocks, RmsNormParamGrad<T, ComputeType>, block_size, 0));
  int waves = 1;
  int dev;
  OF_CUDA_CHECK(cudaGetDevice(&dev));
  int sm_count;
  OF_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev));
  int num_blocks = max_active_blocks * sm_count * waves;
  int grid_dim_y = std::min(max_grid_dim_y, static_cast<int>(num_blocks / grid_dim_x));
  return std::max(grid_dim_y, 1);
}

template<typename T, bool affine>
void RmsNormBackwardGpu(ep::Stream* stream, const int64_t nrows, const int64_t ncols,
                        const T* dy_dptr, const T* x_dptr, const user_op::Tensor* inv_rms,
                        const T* weight_dptr, T* dx_ptr) {
  using ComputeType = typename cuda::layer_norm::DefaultComputeType<T>::type;
  layer_norm::DirectLoad<T, ComputeType> load_x(x_dptr, ncols);
  AffineLoad<T, ComputeType, affine> load_dy(dy_dptr, weight_dptr, ncols);
  layer_norm::DirectStore<ComputeType, T> store(dx_ptr, ncols);
  OF_CUDA_CHECK((rms_norm::DispatchRmsNormGrad<decltype(load_x), decltype(load_dy), decltype(store),
                                               ComputeType>(
      stream->As<ep::CudaStream>()->cuda_stream(), nrows, ncols, load_x, load_dy, store,
      inv_rms->dptr<ComputeType>())));
}

template<typename T>
void DispatchRmsNormBackwardGpu(ep::Stream* stream, const int64_t nrows, const int64_t ncols,
                                const T* dy_dptr, const T* x_dptr, const user_op::Tensor* inv_rms,
                                const T* weight_dptr, T* dx_dptr) {
  if (weight_dptr) {
    RmsNormBackwardGpu<T, true>(stream, nrows, ncols, dy_dptr, x_dptr, inv_rms, weight_dptr,
                                dx_dptr);
  } else {
    RmsNormBackwardGpu<T, false>(stream, nrows, ncols, dy_dptr, x_dptr, inv_rms, weight_dptr,
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
    const double eps = ctx->Attr<double>("epsilon");
    const Shape& normalized_shape = ctx->Attr<Shape>("normalized_shape");
    const int64_t ncols = normalized_shape.elem_cnt();
    const int64_t nrows = inv_rms->shape_view().elem_cnt();
    CHECK_EQ(x->shape_view().elem_cnt(), ncols * nrows);

    const T* weight_dptr = nullptr;
    if (ctx->has_input("weight", 0)) {
      const auto* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
      CHECK_EQ(weight->shape_view().elem_cnt(), ncols);
      weight_dptr = weight->dptr<T>();
    }
    rms_norm::DispatchRmsNormForwardGpu<T>(ctx->stream(), nrows, ncols, eps, x->dptr<T>(),
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
    const int64_t nrows = inv_rms->shape_view().elem_cnt();
    const int64_t ncols = x->shape_view().elem_cnt() / nrows;
    const T* weight_dptr = nullptr;
    if (ctx->has_input("weight", 0)) {
      const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
      CHECK_EQ(ncols, weight->shape_view().elem_cnt());
      weight_dptr = weight->dptr<T>();
    }
    rms_norm::DispatchRmsNormBackwardGpu<T>(ctx->stream(), nrows, ncols, dy->dptr<T>(),
                                            x->dptr<T>(), inv_rms, weight_dptr, dx->mut_dptr<T>());
  };
};

#define REGISTER_RMS_NORM_GRAD_CUDA_KERNEL(dtype)                      \
  REGISTER_USER_KERNEL("rms_norm_grad")                                \
      .SetCreateFn<RmsNormGradGpuKernel<dtype>>()                      \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value));

REGISTER_RMS_NORM_GRAD_CUDA_KERNEL(float)
// REGISTER_RMS_NORM_GRAD_CUDA_KERNEL(double)
// REGISTER_RMS_NORM_GRAD_CUDA_KERNEL(half)
// #if CUDA_VERSION >= 11000
// REGISTER_RMS_NORM_GRAD_CUDA_KERNEL(nv_bfloat16)
// #endif

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

    const int64_t nrows = inv_rms->shape_view().elem_cnt();
    const int64_t ncols = weight_grad->shape_view().elem_cnt();

    const int grid_dim_x = (ncols + rms_norm::tile_size - 1) / rms_norm::tile_size;
    const int grid_dim_y = rms_norm::GetGirdDimY<T>(nrows, ncols);
    T* b_weight_grad_dptr = reinterpret_cast<T*>(tmp_buffer->mut_dptr());

    using ComputeType = typename layer_norm::DefaultComputeType<T>::type;
    dim3 grid_dim(grid_dim_x, grid_dim_y);
    dim3 block_dim(32, 32 / rms_norm::num_per_block);
    rms_norm::RmsNormParamGrad<T, ComputeType>
        <<<grid_dim, block_dim, 0, ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
            nrows, ncols, dy->dptr<T>(), x->dptr<T>(), inv_rms->dptr<ComputeType>(),
            b_weight_grad_dptr);
    const int32_t m = ncols;
    const int32_t n = 1;
    const int32_t k = grid_dim_y;
    const DataType data_type = dy->data_type();
    auto fill = ep::primitive::NewPrimitive<ep::primitive::FillFactory>(
        ctx->stream()->device_type(), data_type);
    CHECK(fill);
    const size_t b_weight_grad_size = grid_dim_y * ncols * sizeof(T);
    T* ones_vec_dptr = reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>() + b_weight_grad_size);
    fill->Launch(ctx->stream(), ones_vec_dptr, 1.0, grid_dim_y);
    auto matmul = ep::primitive::NewPrimitive<ep::primitive::MatmulFactory>(
        ctx->stream()->device_type(), data_type, ep::primitive::BlasTransposeType::T,
        ep::primitive::BlasTransposeType::N);
    CHECK(matmul);
    matmul->Launch(ctx->stream(), m, n, k, 1.0, b_weight_grad_dptr, ones_vec_dptr, 0.0,
                   weight_grad->mut_dptr());
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
        const int64_t nrows = b_shape.elem_cnt();                                       \
        const int64_t ncols = shape.elem_cnt() / nrows;                                 \
        const int grid_dim_y = rms_norm::GetGirdDimY<dtype>(nrows, ncols);              \
        size_t tmp_buffer_size = (grid_dim_y * ncols + grid_dim_y) * sizeof(dtype);     \
        return tmp_buffer_size;                                                         \
      });

REGISTER_RMS_NORM_PARAM_GRAD_GPU_KERNEL(float)
REGISTER_RMS_NORM_PARAM_GRAD_GPU_KERNEL(double)
REGISTER_RMS_NORM_PARAM_GRAD_GPU_KERNEL(half)
#if CUDA_VERSION >= 11000
REGISTER_RMS_NORM_PARAM_GRAD_GPU_KERNEL(nv_bfloat16)
#endif

}  // namespace cuda
}  // namespace oneflow
