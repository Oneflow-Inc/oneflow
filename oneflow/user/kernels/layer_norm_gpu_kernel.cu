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

#include "oneflow/core/device/cudnn_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/cuda/atomic.cuh"
#include <cub/cub.cuh>
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/ep/include/primitive/fill.h"
#include "oneflow/core/ep/include/primitive/matmul.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/cuda/layer_norm.cuh"
#if CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#endif  // CUDA_VERSION >= 11000

namespace oneflow {

namespace {

template<typename SRC, typename DST, bool do_scale, bool do_center>
struct AffineStore {
  AffineStore(DST* y, int64_t row_size, const DST* gamma, const DST* beta)
      : y(y), row_size(row_size), gamma(gamma), beta(beta) {}
  template<int N>
  __device__ void store(const SRC* src, int64_t row, int64_t col) {
    cuda::layer_norm::Pack<DST, N> y_pack;
    cuda::layer_norm::Pack<DST, N> gamma_pack;
    cuda::layer_norm::Pack<DST, N> beta_pack;
    const int64_t offset = (row * row_size + col) / N;
    const int64_t gamma_offset = col / N;
    if (do_scale) {
      gamma_pack.storage =
          *(reinterpret_cast<const cuda::layer_norm::PackType<DST, N>*>(gamma) + gamma_offset);
    } else {
#pragma unroll
      for (int i = 0; i < N; ++i) { gamma_pack.elem[i] = static_cast<DST>(1.f); }
    }
    if (do_center) {
      beta_pack.storage =
          *(reinterpret_cast<const cuda::layer_norm::PackType<DST, N>*>(beta) + gamma_offset);
    } else {
#pragma unroll
      for (int i = 0; i < N; ++i) { beta_pack.elem[i] = static_cast<DST>(0.f); }
    }
#pragma unroll
    for (int i = 0; i < N; ++i) {
      DST normalized_i = static_cast<DST>(src[i]);
      if (do_scale || do_center) {
        y_pack.elem[i] = normalized_i * gamma_pack.elem[i] + beta_pack.elem[i];
      } else {
        y_pack.elem[i] = normalized_i;
      }
    }
    *(reinterpret_cast<cuda::layer_norm::PackType<DST, N>*>(y) + offset) = y_pack.storage;
  }
  DST* y;
  int64_t row_size;
  const DST* gamma;
  const DST* beta;
};

template<typename SRC, typename DST, bool do_scale>
struct ScaleLoad {
  using LoadType = DST;
  ScaleLoad(const SRC* src, const SRC* gamma, int64_t row_size)
      : src(src), gamma(gamma), row_size(row_size) {}
  template<int N>
  __device__ void load(DST* dst, int64_t row, int64_t col) const {
    cuda::layer_norm::Pack<SRC, N> src_pack;
    cuda::layer_norm::Pack<SRC, N> gamma_pack;
    const int64_t offset = (row * row_size + col) / N;
    const int64_t gamma_offset = col / N;
    src_pack.storage = *(reinterpret_cast<const cuda::layer_norm::PackType<SRC, N>*>(src) + offset);
    if (do_scale) {
      gamma_pack.storage =
          *(reinterpret_cast<const cuda::layer_norm::PackType<SRC, N>*>(gamma) + gamma_offset);
    } else {
#pragma unroll
      for (int i = 0; i < N; ++i) { gamma_pack.elem[i] = static_cast<SRC>(1.f); }
    }
#pragma unroll
    for (int i = 0; i < N; ++i) {
      dst[i] = static_cast<DST>(src_pack.elem[i] * gamma_pack.elem[i]);
    }
  }
  const SRC* src;
  const SRC* gamma;
  int64_t row_size;
};

template<typename SRC, typename DST, bool do_add>
struct AddStore {
  AddStore(const DST* add_to_output, DST* dst, int64_t row_size)
      : add_to_output(add_to_output), dst(dst), row_size(row_size) {}
  template<int N>
  __device__ void store(const SRC* src, int64_t row, int64_t col) {
    cuda::layer_norm::Pack<DST, N> add_to_output_pack;
    cuda::layer_norm::Pack<DST, N> dst_pack;
    const int64_t offset = (row * row_size + col) / N;
    if (do_add) {
      add_to_output_pack.storage =
          *(reinterpret_cast<const cuda::layer_norm::PackType<DST, N>*>(add_to_output) + offset);
    }
#pragma unroll
    for (int i = 0; i < N; ++i) {
      if (do_add) {
        dst_pack.elem[i] = static_cast<DST>(src[i]) + add_to_output_pack.elem[i];
      } else {
        dst_pack.elem[i] = static_cast<DST>(src[i]);
      }
    }
    *(reinterpret_cast<cuda::layer_norm::PackType<DST, N>*>(dst) + offset) = dst_pack.storage;
  }
  const DST* add_to_output;
  DST* dst;
  int64_t row_size;
};

template<typename T>
__inline__ __device__ T WarpReduce(T val) {
  for (int mask = 16; mask > 0; mask /= 2) { val += __shfl_down_sync(0xffffffff, val, mask); }
  return val;
}

constexpr int tile_size = 32;
constexpr int num_per_block = 4;
constexpr int block_dim_x = 32;
constexpr int block_dim_y = 32 / num_per_block;

template<typename T, typename ComputeType>
__global__ void LayerNormParamGrad(int rows, int cols, const T* __restrict__ dy,
                                   const T* __restrict__ x, const ComputeType* __restrict__ mean,
                                   const ComputeType* __restrict__ inv_var,
                                   T* __restrict__ tmp_gamma_diff, T* __restrict__ tmp_beta_diff) {
  __shared__ ComputeType dgamma[32][33];
  __shared__ ComputeType dbeta[32][33];
  ComputeType dgamma_sum[num_per_block];
  ComputeType dbeta_sum[num_per_block];
#pragma unroll
  for (int index = 0; index < num_per_block; ++index) {
    dgamma_sum[index] = 0;
    dbeta_sum[index] = 0;
  }
  const int col_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (col_id < cols) {
    for (int i = blockIdx.y * tile_size + threadIdx.y; i < rows; i += tile_size * gridDim.y) {
#pragma unroll
      for (int index = 0; index < num_per_block; ++index) {
        int row_id = i + index * blockDim.y;
        if (row_id < rows) {
          int offset = row_id * cols + col_id;
          const ComputeType dy_val = static_cast<ComputeType>(dy[offset]);
          const ComputeType x_val = static_cast<ComputeType>(x[offset]);
          const ComputeType mean_val = mean[row_id];
          const ComputeType inv_var_val = inv_var[row_id];
          dgamma_sum[index] += dy_val * (x_val - mean_val) * inv_var_val;
          dbeta_sum[index] += dy_val;
        }
      }
    }
  }
#pragma unroll
  for (int index = 0; index < num_per_block; ++index) {
    dgamma[index * blockDim.y + threadIdx.y][threadIdx.x] = dgamma_sum[index];
    dbeta[index * blockDim.y + threadIdx.y][threadIdx.x] = dbeta_sum[index];
  }
  __syncthreads();
#pragma unroll
  for (int index = 0; index < num_per_block; ++index) {
    const int col_id = blockIdx.x * blockDim.x + threadIdx.y + index * blockDim.y;
    if (col_id < cols) {
      ComputeType gamma_sum = dgamma[threadIdx.x][threadIdx.y + index * blockDim.y];
      ComputeType beta_sum = dbeta[threadIdx.x][threadIdx.y + index * blockDim.y];
      ComputeType global_dgamma = WarpReduce<ComputeType>(gamma_sum);
      ComputeType global_dbeta = WarpReduce<ComputeType>(beta_sum);
      if (threadIdx.x == 0) {
        const int offset = blockIdx.y * cols + col_id;
        tmp_gamma_diff[offset] = global_dgamma;
        tmp_beta_diff[offset] = global_dbeta;
      }
    }
  }
}

template<typename T>
int GetGirdDimY(const int64_t num_instances, const int64_t norm_size) {
  using ComputeType = typename cuda::layer_norm::DefaultComputeType<T>::type;
  const int grid_dim_x = (norm_size + tile_size - 1) / tile_size;
  const int max_grid_dim_y = (num_instances + tile_size - 1) / tile_size;
  const int block_size = block_dim_x * block_dim_y;
  int max_active_blocks = 0;
  OF_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_blocks, LayerNormParamGrad<T, ComputeType>, block_size, 0));
  int waves = 1;
  int dev;
  OF_CUDA_CHECK(cudaGetDevice(&dev));
  int sm_count;
  OF_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev));
  int num_blocks = max_active_blocks * sm_count * waves;
  int grid_dim_y = std::min(max_grid_dim_y, static_cast<int>(num_blocks / grid_dim_x));
  return std::max(grid_dim_y, 1);
}

template<typename T, bool do_scale, bool do_center>
void LayerNormForwardGpu(ep::Stream* stream, const int64_t num_instances, const int64_t norm_size,
                         const double epsilon, const T* x_ptr, const T* gamma_ptr,
                         const T* beta_ptr, T* y_ptr, user_op::Tensor* mean,
                         user_op::Tensor* inv_variance) {
  using ComputeType = typename cuda::layer_norm::DefaultComputeType<T>::type;
  cuda::layer_norm::DirectLoad<T, T> load(x_ptr, norm_size);
  AffineStore<ComputeType, T, do_scale, do_center> store(y_ptr, norm_size, gamma_ptr, beta_ptr);
  cuda::layer_norm::DispatchLayerNorm<decltype(load), decltype(store), ComputeType>(
      stream->As<ep::CudaStream>()->cuda_stream(), load, store, num_instances, norm_size, epsilon,
      mean->mut_dptr<ComputeType>(), inv_variance->mut_dptr<ComputeType>());
}

template<typename T>
void DispatchLayerNormForwardGpu(ep::Stream* stream, const int64_t num_instances,
                                 const int64_t norm_size, const double epsilon, const T* x_ptr,
                                 const T* gamma_ptr, const T* beta_ptr, T* y_ptr,
                                 user_op::Tensor* mean, user_op::Tensor* inv_variance) {
  if (gamma_ptr != nullptr && beta_ptr != nullptr) {
    LayerNormForwardGpu<T, true, true>(stream, num_instances, norm_size, epsilon, x_ptr, gamma_ptr,
                                       beta_ptr, y_ptr, mean, inv_variance);
  } else if (gamma_ptr != nullptr && beta_ptr == nullptr) {
    LayerNormForwardGpu<T, true, false>(stream, num_instances, norm_size, epsilon, x_ptr, gamma_ptr,
                                        beta_ptr, y_ptr, mean, inv_variance);
  } else if (gamma_ptr == nullptr && beta_ptr != nullptr) {
    LayerNormForwardGpu<T, false, true>(stream, num_instances, norm_size, epsilon, x_ptr, gamma_ptr,
                                        beta_ptr, y_ptr, mean, inv_variance);
  } else {
    LayerNormForwardGpu<T, false, false>(stream, num_instances, norm_size, epsilon, x_ptr,
                                         gamma_ptr, beta_ptr, y_ptr, mean, inv_variance);
  }
}

template<typename T, bool do_scale, bool do_add>
void LayerNormBackwardGpu(ep::Stream* stream, const int64_t num_instances, const int64_t norm_size,
                          const T* dy_ptr, const T* x_ptr, const user_op::Tensor* mean,
                          const user_op::Tensor* inv_variance, const T* gamma_ptr,
                          const T* add_to_output_ptr, T* dx_ptr) {
  using ComputeType = typename cuda::layer_norm::DefaultComputeType<T>::type;
  cuda::layer_norm::DirectLoad<T, T> load_x(x_ptr, norm_size);
  ScaleLoad<T, T, do_scale> load_scaled_dy(dy_ptr, gamma_ptr, norm_size);
  AddStore<ComputeType, T, do_add> store(add_to_output_ptr, dx_ptr, norm_size);
  OF_CUDA_CHECK((cuda::layer_norm::DispatchLayerNormGrad<decltype(load_x), decltype(load_scaled_dy),
                                                         decltype(store), ComputeType>(
      stream->As<ep::CudaStream>()->cuda_stream(), load_x, load_scaled_dy, store,
      mean->dptr<ComputeType>(), inv_variance->dptr<ComputeType>(), num_instances, norm_size)));
}

template<typename T, bool do_scale>
void DispatchLayerNormBackwardDoAdd(ep::Stream* stream, const int64_t num_instances,
                                    const int64_t norm_size, const T* dy_ptr, const T* x_ptr,
                                    const user_op::Tensor* mean,
                                    const user_op::Tensor* inv_variance, const T* gamma_ptr,
                                    const T* add_to_output_ptr, T* dx_ptr) {
  if (add_to_output_ptr != nullptr) {
    LayerNormBackwardGpu<T, do_scale, true>(stream, num_instances, norm_size, dy_ptr, x_ptr, mean,
                                            inv_variance, gamma_ptr, add_to_output_ptr, dx_ptr);
  } else {
    LayerNormBackwardGpu<T, do_scale, false>(stream, num_instances, norm_size, dy_ptr, x_ptr, mean,
                                             inv_variance, gamma_ptr, add_to_output_ptr, dx_ptr);
  }
}

template<typename T>
void LaunchLayerNormBackward(ep::Stream* stream, const int64_t num_instances,
                             const int64_t norm_size, const T* dy_ptr, const T* x_ptr,
                             const user_op::Tensor* mean, const user_op::Tensor* inv_variance,
                             const T* gamma_ptr, const T* add_to_output_ptr, T* dx_ptr) {
  if (gamma_ptr != nullptr) {
    DispatchLayerNormBackwardDoAdd<T, true>(stream, num_instances, norm_size, dy_ptr, x_ptr, mean,
                                            inv_variance, gamma_ptr, add_to_output_ptr, dx_ptr);
  } else {
    DispatchLayerNormBackwardDoAdd<T, false>(stream, num_instances, norm_size, dy_ptr, x_ptr, mean,
                                             inv_variance, gamma_ptr, add_to_output_ptr, dx_ptr);
  }
}

}  // namespace

template<typename T>
class LayerNormGpuKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  LayerNormGpuKernel() = default;
  ~LayerNormGpuKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* mean = ctx->Tensor4ArgNameAndIndex("mean", 0);
    user_op::Tensor* inv_variance = ctx->Tensor4ArgNameAndIndex("inv_variance", 0);
    const double epsilon = ctx->Attr<double>("epsilon");
    CHECK_GE(epsilon, CUDNN_BN_MIN_EPSILON);
    const int64_t num_instances = mean->shape_view().elem_cnt();
    const int64_t norm_size = x->shape_view().elem_cnt() / num_instances;
    const T* gamma_ptr = nullptr;
    const T* beta_ptr = nullptr;
    if (ctx->has_input("gamma", 0)) {
      const user_op::Tensor* gamma = ctx->Tensor4ArgNameAndIndex("gamma", 0);
      gamma_ptr = gamma->dptr<T>();
      CHECK_EQ(gamma->shape_view().elem_cnt(), norm_size);
    }
    if (ctx->has_input("beta", 0)) { beta_ptr = ctx->Tensor4ArgNameAndIndex("beta", 0)->dptr<T>(); }
    DispatchLayerNormForwardGpu<T>(ctx->stream(), num_instances, norm_size, epsilon, x->dptr<T>(),
                                   gamma_ptr, beta_ptr, y->mut_dptr<T>(), mean, inv_variance);
  };
};

#define REGISTER_LAYER_NORM_CUDA_KERNEL(dtype)                         \
  REGISTER_USER_KERNEL("layer_norm")                                   \
      .SetCreateFn<LayerNormGpuKernel<dtype>>()                        \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_LAYER_NORM_CUDA_KERNEL(float)
REGISTER_LAYER_NORM_CUDA_KERNEL(double)
REGISTER_LAYER_NORM_CUDA_KERNEL(half)
#if CUDA_VERSION >= 11000
REGISTER_LAYER_NORM_CUDA_KERNEL(nv_bfloat16)
#endif

template<typename T>
class LayerNormGradGpuKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  LayerNormGradGpuKernel() = default;
  ~LayerNormGradGpuKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* mean = ctx->Tensor4ArgNameAndIndex("mean", 0);
    const user_op::Tensor* inv_variance = ctx->Tensor4ArgNameAndIndex("inv_variance", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const int64_t num_instances = mean->shape_view().elem_cnt();
    const int64_t norm_size = x->shape_view().elem_cnt() / num_instances;
    const T* gamma_ptr = nullptr;
    if (ctx->has_input("gamma", 0)) {
      gamma_ptr = ctx->Tensor4ArgNameAndIndex("gamma", 0)->dptr<T>();
    }
    const T* add_to_output_ptr = nullptr;
    if (ctx->has_input("_add_to_output", 0)) {
      const user_op::Tensor* add_to_output = ctx->Tensor4ArgNameAndIndex("_add_to_output", 0);
      CHECK_EQ(add_to_output->data_type(), dx->data_type());
      CHECK_EQ(add_to_output->shape_view(), dx->shape_view());
      add_to_output_ptr = add_to_output->dptr<T>();
    }
    LaunchLayerNormBackward<T>(ctx->stream(), num_instances, norm_size, dy->dptr<T>(), x->dptr<T>(),
                               mean, inv_variance, gamma_ptr, add_to_output_ptr, dx->mut_dptr<T>());
  };
};

#define REGISTER_LAYER_NORM_GRAD_CUDA_KERNEL(dtype)                                        \
  REGISTER_USER_KERNEL("layer_norm_grad")                                                  \
      .SetCreateFn<LayerNormGradGpuKernel<dtype>>()                                        \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                     \
                       && (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value))    \
      .SetInplaceProposalFn(                                                               \
          [](const user_op::InferContext& ctx,                                             \
             const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {       \
            if (ctx.has_input("_add_to_output", 0)) {                                      \
              OF_RETURN_IF_ERROR(AddInplaceArgPairFn("dx", 0, "_add_to_output", 0, true)); \
            }                                                                              \
            return Maybe<void>::Ok();                                                      \
          });

REGISTER_LAYER_NORM_GRAD_CUDA_KERNEL(float)
REGISTER_LAYER_NORM_GRAD_CUDA_KERNEL(double)
REGISTER_LAYER_NORM_GRAD_CUDA_KERNEL(half)
#if CUDA_VERSION >= 11000
REGISTER_LAYER_NORM_GRAD_CUDA_KERNEL(nv_bfloat16)
#endif

template<typename T>
class LayerNormParamGradGpuKernel final : public user_op::OpKernel,
                                          public user_op::CudaGraphSupport {
 public:
  LayerNormParamGradGpuKernel() = default;
  ~LayerNormParamGradGpuKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* mean = ctx->Tensor4ArgNameAndIndex("mean", 0);
    const user_op::Tensor* inv_variance = ctx->Tensor4ArgNameAndIndex("inv_variance", 0);
    const int64_t num_instances = mean->shape_view().elem_cnt();
    const int64_t norm_size = x->shape_view().elem_cnt() / num_instances;
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const DataType data_type = dy->data_type();
    const int grid_dim_x = (norm_size + tile_size - 1) / tile_size;
    const int grid_dim_y = GetGirdDimY<T>(num_instances, norm_size);
    const size_t tmp_gamma_diff_size = grid_dim_y * norm_size * sizeof(T);
    T* tmp_gamma_diff_ptr = reinterpret_cast<T*>(tmp_buffer->mut_dptr());
    T* tmp_beta_diff_ptr = reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>() + tmp_gamma_diff_size);
    T* reduce_buf_ptr =
        reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>() + 2 * tmp_gamma_diff_size);
    using ComputeType = typename cuda::layer_norm::DefaultComputeType<T>::type;
    LayerNormParamGrad<T, ComputeType><<<dim3(grid_dim_x, grid_dim_y), dim3(32, 32 / num_per_block),
                                         0, ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
        num_instances, norm_size, dy->dptr<T>(), x->dptr<T>(), mean->dptr<ComputeType>(),
        inv_variance->dptr<ComputeType>(), tmp_gamma_diff_ptr, tmp_beta_diff_ptr);
    const int32_t m = norm_size;
    const int32_t n = 1;
    const int32_t k = grid_dim_y;
    std::unique_ptr<ep::primitive::Fill> fill =
        ep::primitive::NewPrimitive<ep::primitive::FillFactory>(ctx->stream()->device_type(),
                                                                data_type);
    CHECK(fill);
    fill->Launch(ctx->stream(), reduce_buf_ptr, 1.0, grid_dim_y);
    std::unique_ptr<ep::primitive::Matmul> matmul =
        ep::primitive::NewPrimitive<ep::primitive::MatmulFactory>(
            ctx->stream()->device_type(), data_type, ep::primitive::BlasTransposeType::T,
            ep::primitive::BlasTransposeType::N);
    CHECK(matmul);
    if (ctx->has_output("gamma_diff", 0)) {
      user_op::Tensor* gamma_diff = ctx->Tensor4ArgNameAndIndex("gamma_diff", 0);
      matmul->Launch(ctx->stream(), m, n, k, 1.0, tmp_gamma_diff_ptr, reduce_buf_ptr, 0.0,
                     gamma_diff->mut_dptr());
    }
    if (ctx->has_output("beta_diff", 0)) {
      user_op::Tensor* beta_diff = ctx->Tensor4ArgNameAndIndex("beta_diff", 0);
      matmul->Launch(ctx->stream(), m, n, k, 1.0, tmp_beta_diff_ptr, reduce_buf_ptr, 0.0,
                     beta_diff->mut_dptr());
    }
  };
};

#define REGISTER_LAYER_NORM_PARAM_GRAD_GPU_KERNEL(dtype)                                    \
  REGISTER_USER_KERNEL("layer_norm_param_grad")                                             \
      .SetCreateFn<LayerNormParamGradGpuKernel<dtype>>()                                    \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                      \
                       && (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value))     \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                   \
        const int64_t begin_params_axis = ctx->Attr<int64_t>("begin_params_axis");          \
        const bool has_gamma_diff = ctx->has_output("gamma_diff", 0);                       \
        const bool has_beta_diff = ctx->has_output("beta_diff", 0);                         \
        const auto& dy = ctx->InputTensorDesc("dy", 0);                                     \
        const int64_t num_instances = dy.shape().Count(0, begin_params_axis);               \
        const int64_t norm_size = dy.shape().Count(begin_params_axis);                      \
        const int grid_dim_y = GetGirdDimY<dtype>(num_instances, norm_size);                \
        size_t tmp_buffer_size = (2 * grid_dim_y * norm_size + grid_dim_y) * sizeof(dtype); \
        return tmp_buffer_size;                                                             \
      });

REGISTER_LAYER_NORM_PARAM_GRAD_GPU_KERNEL(float)
REGISTER_LAYER_NORM_PARAM_GRAD_GPU_KERNEL(double)
REGISTER_LAYER_NORM_PARAM_GRAD_GPU_KERNEL(half)
#if CUDA_VERSION >= 11000
REGISTER_LAYER_NORM_PARAM_GRAD_GPU_KERNEL(nv_bfloat16)
#endif

}  // namespace oneflow
