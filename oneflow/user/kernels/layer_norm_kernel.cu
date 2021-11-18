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
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/cuda/layer_norm.cuh"
#include "oneflow/core/ep/include/primitive/fill.h"

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
    const int64_t offset = row * row_size + col;
    if (do_scale) {
      gamma_pack.storage =
          *reinterpret_cast<const cuda::layer_norm::PackType<DST, N>*>(gamma + col);
    } else {
#pragma unroll
      for (int i = 0; i < N; ++i) { gamma_pack.elem[i] = 1; }
    }
    if (do_center) {
      beta_pack.storage = *reinterpret_cast<const cuda::layer_norm::PackType<DST, N>*>(beta + col);
    } else {
#pragma unroll
      for (int i = 0; i < N; ++i) { beta_pack.elem[i] = 0; }
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
    *reinterpret_cast<cuda::layer_norm::PackType<DST, N>*>(y + offset) = y_pack.storage;
  }
  DST* y;
  int64_t row_size;
  const DST* gamma;
  const DST* beta;
};

template<typename SRC, typename DST, bool do_scale>
struct ScaleLoad {
  ScaleLoad(const SRC* src, const SRC* gamma, int64_t row_size)
      : src(src), gamma(gamma), row_size(row_size) {}
  template<int N>
  __device__ void load(DST* dst, int64_t row, int64_t col) const {
    Pack<SRC, N> src_pack;
    Pack<SRC, N> gamma_pack;
    const int64_t offset = row * row_size + col;
    src_pack.storage = *reinterpret_cast<const PackType<SRC, N>*>(src + offset);
    if (do_scale) {
      gamma_pack.storage =
          *reinterpret_cast<const cuda::layer_norm::PackType<DST, N>*>(gamma + col);
    } else {
#pragma unroll
      for (int i = 0; i < N; ++i) { gamma_pack.elem[i] = 1; }
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

}  // namespace

template<typename T, bool do_scale, bool do_center>
void LayerNormForwardGpu(DeviceCtx* ctx, const int num_instances, const int norm_size,
                         const double epsilon, const T* x_ptr, const T* gamma_ptr,
                         const T* beta_ptr, T* y_ptr, user_op::Tensor* mean,
                         user_op::Tensor* inv_variance) {
  using ComputeType = typename cuda::layer_norm::DefaultComputeType<T>::type;
  cuda::layer_norm::DirectLoad<T, ComputeType> load(x_ptr, norm_size);
  AffineStore<ComputeType, T, do_scale, do_center> store(y_ptr, norm_size, gamma_ptr, beta_ptr);
  cuda::layer_norm::DispatchLayerNorm<decltype(load), decltype(store), ComputeType>(
      ctx->cuda_stream(), load, store, num_instances, norm_size, epsilon,
      mean->mut_dptr<ComputeType>(), inv_variance->mut_dptr<ComputeType>());
}

template<typename T>
void LaunchLayerNormForward(DeviceCtx* ctx, const int num_instances, const int norm_size,
                            const double epsilon, const T* x_ptr, const T* gamma_ptr,
                            const T* beta_ptr, T* y_ptr, user_op::Tensor* mean,
                            user_op::Tensor* inv_variance) {
  if (gamma_ptr != nullptr && beta_ptr != nullptr) {
    LayerNormForwardGpu<T, true, true>(ctx, num_instances, norm_size, epsilon, x_ptr, gamma_ptr,
                                       beta_ptr, y_ptr, mean, inv_variance);
  } else if (gamma_ptr != nullptr && beta_ptr == nullptr) {
    LayerNormForwardGpu<T, true, false>(ctx, num_instances, norm_size, epsilon, x_ptr, gamma_ptr,
                                        beta_ptr, y_ptr, mean, inv_variance);
  } else if (gamma_ptr == nullptr && beta_ptr != nullptr) {
    LayerNormForwardGpu<T, false, true>(ctx, num_instances, norm_size, epsilon, x_ptr, gamma_ptr,
                                        beta_ptr, y_ptr, mean, inv_variance);
  } else {
    LayerNormForwardGpu<T, false, false>(ctx, num_instances, norm_size, epsilon, x_ptr, gamma_ptr,
                                         beta_ptr, y_ptr, mean, inv_variance);
  }
}

template<typename T, bool do_scale>
void LayerNormBackwardGpu(DeviceCtx* ctx, const int num_instances, const int norm_size,
                          const T* dy_ptr, const T* x_ptr, user_op::Tensor* mean,
                          user_op::Tensor* inv_variance, const T* gamma_ptr, T* dx_ptr) {
  using ComputeType = typename cuda::layer_norm::DefaultComputeType<T>::type;
  cuda::layer_norm::DirectLoad<T, ComputeType> load_x(x->dptr<T>(), norm_size);
  cuda::layer_norm::ScaleLoad<T, ComputeType, do_scale> load_dy(dy->dptr<T>(), norm_size);
  cuda::layer_norm::DirectStore<ComputeType, T> store(dx->mut_dptr<T>(), norm_size);
  OF_CUDA_CHECK((cuda::layer_norm::DispatchLayerNormGrad<decltype(load_x), decltype(load_dy),
                                                         decltype(store), ComputeType>(
      ctx->cuda_stream(), load_x, load_dy, store, mean->dptr<ComputeType>(),
      inv_variance->dptr<ComputeType>(), num_instances, norm_size)));
}

template<typename T>
void LaunchLayerNormBackward(DeviceCtx* ctx, const int num_instances, const int norm_size,
                             const T* dy_ptr, const T* x_ptr, user_op::Tensor* mean,
                             user_op::Tensor* inv_variance, const T* gamma_ptr, T* dx_ptr) {
  if (gamma_ptr != nullptr) {
    LayerNormBackwardGpu<T, true>(ctx, num_instances, norm_size, dy_ptr, x_ptr, mean, inv_variance,
                                  gamma_ptr, dx_ptr);
  } else {
    LayerNormBackwardGpu<T, false>(ctx, num_instances, norm_size, epsilon, x_ptr, mean,
                                   inv_variance, gamma_ptr, dx_ptr);
  }
}

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
    const int32_t num_instances = mean->shape().elem_cnt();
    const int32_t norm_size = x->shape().elem_cnt() / rows;
    const T* gamma_ptr = nullptr;
    const T* beta_ptr = nullptr;
    if (ctx->has_input("gamma", 0)) {
      gamma_ptr = ctx->Tensor4ArgNameAndIndex("gamma", 0)->dptr<T>();
    }
    if (ctx->has_input("beta", 0)) { beta_ptr = ctx->Tensor4ArgNameAndIndex("beta", 0)->dptr<T>(); }
    LaunchLayerNormForward<T>(ctx->device_ctx(), num_instances, norm_size, epsilon, x->dptr<T>(),
                              gamma_ptr, beta_ptr, y->mut_dptr<T>(), mean, inv_variance);
  };
};

#define REGISTER_LAYER_NORM_GPU_KERNEL(dtype)                         \
  REGISTER_USER_KERNEL("layer_norm")                                  \
      .SetCreateFn<LayerNormGpuKernel<dtype>>()                       \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kGPU) \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_LAYER_NORM_GPU_KERNEL(float)
REGISTER_LAYER_NORM_GPU_KERNEL(double)
REGISTER_LAYER_NORM_GPU_KERNEL(half)

template<typename T, typename ComputeType, typename I>
__global__ void LayerNormParamGradImpl(const I n, const I instance_size, const T* dy, const T* x,
                                       const ComputeType* mean, const ComputeType* inv_variance,
                                       const T* gamma, T* tmp_gamma_diff, T* tmp_beta_diff) {
  extern __shared__ __align__(sizeof(double)) unsigned char bw_shared_buf[];
  ComputeType* gamma_diff_sum_buf = reinterpret_cast<ComputeType*>(bw_shared_buf);
  ComputeType* beta_diff_sum_buf = gamma_diff_sum_buf + instance_size;
  const I tid = threadIdx.x;
  for (I col_id = tid; col_id < instance_size; col_id += blockDim.x) {
    gamma_diff_sum_buf[col_id] = 0;
    beta_diff_sum_buf[col_id] = 0;
  }
  __syncthreads();
  CUDA_1D_KERNEL_LOOP_T(I, i, n) {
    const I row_id = i / instance_size;
    const I col_id = i - row_id * instance_size;
    ComputeType dy_val = static_cast<ComputeType>(dy[i]);
    ComputeType normalized_val =
        (static_cast<ComputeType>(x[i]) - mean[row_id]) * inv_variance[row_id];
    cuda::atomic::Add(&gamma_diff_sum_buf[col_id], dy_val * normalized_val);
    cuda::atomic::Add(&beta_diff_sum_buf[col_id], dy_val);
  }
  __syncthreads();
  for (I col_id = tid; col_id < instance_size; col_id += blockDim.x) {
    const I offset = blockIdx.x * instance_size + elem_id;
    tmp_gamma_diff[offset] = static_cast<T>(gamma_diff_sum_buf[col_id]);
    tmp_beta_diff[offset] = static_cast<T>(beta_diff_sum_buf[col_id]);
  }
}

constexpr int64_t kLayerNormParamGradGpuBlockSize = 512;

int64_t GetLayerNormParamGradNumBlocks(const int64_t elem_cnt) {
  return std::min(static_cast<int>((elem_cnt + kLayerNormParamGradGpuBlockSize - 1)
                                   / kLayerNormParamGradGpuBlockSize),
                  256);
}

template<typename T>
void LaunchLayerNormParamBackward(DeviceCtx* ctx, const int64_t elem_cnt, const int64_t norm_size,
                                  const T* dy_ptr, const T* x_ptr, user_op::Tensor* mean,
                                  user_op::Tensor* inv_variance, const T* gamma_ptr,
                                  T* tmp_gamma_diff_ptr, T* tmp_beta_diff_ptr) {
  using ComputeType = typename cuda::layer_norm::DefaultComputeType<T>::type;
  const size_t shared_mempry_size = 2 * norm_size * sizeof(ComputeType);
  if (elem_cnt > static_cast<int64_t>(GetMaxVal<int32_t>() / 2)) {
    LayerNormParamGradImpl<T, ComputeType, int64_t>
        <<<GetLayerNormParamGradNumBlocks(elem_cnt), kLayerNormParamGradGpuBlockSize,
           shared_mempry_size, ctx->device_ctx()->cuda_stream()>>>(
            elem_cnt, norm_size, dy_ptr, x_ptr, mean->dptr<ComputeType>(),
            inv_variance->dptr<ComputeType>(), gamma_ptr, tmp_gamma_diff_ptr, tmp_beta_diff_ptr);
  } else {
    LayerNormParamGradImpl<T, ComputeType, int32_t>
        <<<GetLayerNormParamGradNumBlocks(elem_cnt), kLayerNormParamGradGpuBlockSize,
           shared_mempry_size, ctx->device_ctx()->cuda_stream()>>>(
            static_cast<int32_t>(elem_cnt), static_cast<int32_t>(norm_size), dy_ptr, x_ptr,
            mean->dptr<ComputeType>(), inv_variance->dptr<ComputeType>(), gamma_ptr,
            tmp_gamma_diff_ptr, tmp_beta_diff_ptr);
  }
}

template<typename T>
void DispatchLayerNormParamBackward(DeviceCtx* ctx, const int64_t elem_cnt, const int64_t norm_size,
                                    const T* dy_ptr, const T* x_ptr, user_op::Tensor* mean,
                                    user_op::Tensor* inv_variance, T* tmp_gamma_diff_ptr,
                                    T* tmp_beta_diff_ptr) {
  LaunchLayerNormParamBackward<T>(ctx, elem_cnt, norm_size, dy_ptr, x_ptr, mean, inv_variance,
                                  tmp_gamma_diff_ptr, tmp_beta_diff_ptr);
}

template<>
void DispatchLayerNormParamBackward<float16>(DeviceCtx* ctx, const int64_t elem_cnt,
                                             const int64_t norm_size, const float16* dy_ptr,
                                             const float16* x_ptr, user_op::Tensor* mean,
                                             user_op::Tensor* inv_variance,
                                             float16* tmp_gamma_diff_ptr,
                                             float16* tmp_beta_diff_ptr) {
  LaunchLayerNormParamBackward<half>(ctx, elem_cnt, norm_size, reinterpret_cast<half*>(dy_ptr),
                                     reinterpret_cast<half*>(x_ptr), mean, inv_variance,
                                     reinterpret_cast<half*>(tmp_gamma_diff_ptr),
                                     reinterpret_cast<half*>(tmp_beta_diff_ptr));
}

template<typename T>
class LayerNormGradKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  LayerNormGradKernel() = default;
  ~LayerNormGradKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* mean = ctx->Tensor4ArgNameAndIndex("mean", 0);
    const user_op::Tensor* inv_variance = ctx->Tensor4ArgNameAndIndex("inv_variance", 0);
    const int32_t num_instances = mean->shape().elem_cnt();
    const int32_t norm_size = x->shape().elem_cnt() / num_instances;

    const T* gamma_ptr = nullptr;
    if (ctx->has_input("gamma", 0)) {
      gamma_ptr = ctx->Tensor4ArgNameAndIndex("gamma", 0)->dptr<T>();
    }
    // dx
    if (ctx->has_output("dx", 0)) {
      LaunchLayerNormBackward<T>(ctx->device_ctx(), num_instances, norm_size, dy->dptr<T>(),
                                 x->dptr<T>(), mean, inv_variance, gamma_ptr,
                                 ctx->Tensor4ArgNameAndIndex("dx", 0)->mut_dptr<T>());
    }
    if (ctx->has_output("gamma_diff", 0) && ctx->has_output("beta_diff", 0)
        && max_active_blocks > 0) {
      const int64_t elem_cnt = dy->shape().elem_cnt();
      const int64_t num_blocks = GetLayerNormParamGradNumBlocks(elem_cnt);
      user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
      const size_t tmp_diff_size = GetCudaAlignedSize(num_blocks * norm_size * sizeof(T));
      T* tmp_gamma_diff = tmp_buffer->mut_dptr<T>();
      T* tmp_beta_diff = reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>() + tmp_diff_size);
      T* tmp_reduce_buf = reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>() + 2 * tmp_diff_size);
      DispatchLayerNormParamBackward<T>(ctx->device_ctx(), elem_cnt, norm_size, dy->dptr<T>(),
                                        x->dptr<T>(), mean, inv_variance, tmp_gamma_diff,
                                        tmp_beta_diff);
      NdUtil::ReduceSum(ctx->device_ctx(), Var({1, norm_size}, gamma_diff->mut_dptr<T>()),
                        Val({num_blocks, norm_size}, tmp_gamma_diff),
                        Var({num_blocks, norm_size}, tmp_reduce_buf));
      NdUtil::ReduceSum(ctx->device_ctx(), Var({1, norm_size}, beta_diff->mut_dptr<T>()),
                        Val({num_blocks, norm_size}, tmp_beta_diff),
                        Var({num_blocks, norm_size}, tmp_reduce_buf));
    } else {
      UNIMPLEMENTED();
    }
  };
};

#define REGISTER_LAYER_NORM_GRAD_KERNEL(dtype)                        \
  REGISTER_USER_KERNEL("layer_norm_grad")                             \
      .SetCreateFn<LayerNormGradKernel<dtype>>()                      \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kGPU) \
                       & (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value));

REGISTER_LAYER_NORM_GRAD_KERNEL(float)
REGISTER_LAYER_NORM_GRAD_KERNEL(double)
REGISTER_LAYER_NORM_GRAD_KERNEL(half)

}  // namespace oneflow
