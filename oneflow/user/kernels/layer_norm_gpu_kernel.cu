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
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/cuda/layer_norm.cuh"

namespace oneflow {

namespace {

std::unique_ptr<ep::primitive::Fill> NewFillPrimitive(ep::Stream* stream, DataType data_type) {
  std::unique_ptr<ep::primitive::Fill> fill =
      ep::primitive::NewPrimitive<ep::primitive::FillFactory>(stream->device_type(), data_type);
  CHECK(fill);
  return fill;
}

class LayerNormCudnnBnCtx final {
 public:
  LayerNormCudnnBnCtx(const ShapeView& data_shape, const ShapeView& param_shape,
                      DataType data_type) {
    const int64_t cudnn_c = param_shape.elem_cnt();
    CHECK_EQ(data_shape.elem_cnt() % cudnn_c, 0);
    const int64_t cudnn_w = data_shape.elem_cnt() / cudnn_c;
    CHECK_LT(cudnn_c, GetMaxVal<int32_t>());
    CHECK_LT(cudnn_w, GetMaxVal<int32_t>());
    data_tensor_desc_.reset(new CudnnTensorDesc(CUDNN_TENSOR_NCHW, data_type, 1,
                                                static_cast<int32_t>(cudnn_c), 1,
                                                static_cast<int32_t>(cudnn_w)));
    DataType param_dtype = data_type == DataType::kFloat16 ? DataType::kFloat : data_type;
    param_tensor_desc_.reset(new CudnnTensorDesc(CUDNN_TENSOR_NCHW, param_dtype, 1,
                                                 static_cast<int32_t>(cudnn_c), 1, 1));
#if (CUDNN_VERSION >= 7000)
    mode_ = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
#else
    mode_ = CUDNN_BATCHNORM_SPATIAL;
#endif
  }
  ~LayerNormCudnnBnCtx() = default;

  const cudnnTensorDescriptor_t& data_tensor_desc() const { return data_tensor_desc_->Get(); }
  const cudnnTensorDescriptor_t& param_tensor_desc() const { return param_tensor_desc_->Get(); }
  cudnnBatchNormMode_t mode() const { return mode_; };

 private:
  std::unique_ptr<CudnnTensorDesc> data_tensor_desc_;
  std::unique_ptr<CudnnTensorDesc> param_tensor_desc_;
  cudnnBatchNormMode_t mode_;
};

template<typename SRC, typename DST, bool do_scale, bool do_center>
struct AffineStore {
  AffineStore(DST* normalized, DST* y, int64_t row_size, const DST* gamma, const DST* beta)
      : normalized(normalized), y(y), row_size(row_size), gamma(gamma), beta(beta) {}
  template<int N>
  __device__ void store(const SRC* src, int64_t row, int64_t col) {
    cuda::layer_norm::Pack<DST, N> y_pack;
    cuda::layer_norm::Pack<DST, N> normalized_pack;
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
      if (do_scale) { normalized_pack.elem[i] = normalized_i; }
      if (do_scale || do_center) {
        y_pack.elem[i] = normalized_i * gamma_pack.elem[i] + beta_pack.elem[i];
      } else {
        y_pack.elem[i] = normalized_i;
      }
    }
    *reinterpret_cast<cuda::layer_norm::PackType<DST, N>*>(y + offset) = y_pack.storage;
    if (do_scale) {
      *reinterpret_cast<cuda::layer_norm::PackType<DST, N>*>(normalized + offset) =
          normalized_pack.storage;
    }
  }
  DST* normalized;
  DST* y;
  int64_t row_size;
  const DST* gamma;
  const DST* beta;
};

constexpr int64_t kLayerNormParamGradGpuBlockSize = 512;

int64_t GetLayerNormParamGradBlockSize() { return kLayerNormParamGradGpuBlockSize; }

int64_t GetLayerNormParamGradNumBlocks(const int64_t elem_cnt) {
  return std::min(static_cast<int>((elem_cnt + kLayerNormParamGradGpuBlockSize - 1)
                                   / kLayerNormParamGradGpuBlockSize),
                  256);
}

template<typename T>
int64_t GetParamGradDynamicSharedMemorySize(const int64_t instance_size) {
  return 2 * instance_size * sizeof(T);
}

template<>
int64_t GetParamGradDynamicSharedMemorySize<float16>(const int64_t instance_size) {
  return 2 * instance_size * sizeof(float);
}

template<typename T, typename I>
__global__ void LayerNormParamGradImpl(const I n, const I instance_size, const T* dy,
                                       const T* normalized, const T* gamma, T* gamma_diff,
                                       T* beta_diff, T* normalized_diff) {
  extern __shared__ __align__(sizeof(double)) unsigned char bw_shared_buf[];
  auto* gamma_diff_sum_buf = reinterpret_cast<T*>(bw_shared_buf);
  auto* beta_diff_sum_buf = gamma_diff_sum_buf + instance_size;
  const I tid = threadIdx.x;
  for (I elem_id = tid; elem_id < instance_size; elem_id += blockDim.x) {
    gamma_diff_sum_buf[elem_id] = 0;
    beta_diff_sum_buf[elem_id] = 0;
  }
  __syncthreads();
  CUDA_1D_KERNEL_LOOP_T(I, i, n) {
    const I elem_id = i % instance_size;
    T dy_val = dy[i];
    T normalized_val = normalized[i];
    cuda::atomic::Add(&gamma_diff_sum_buf[elem_id], dy_val * normalized_val);
    cuda::atomic::Add(&beta_diff_sum_buf[elem_id], dy_val);
    T gamma_val = gamma[elem_id];
    normalized_diff[i] = gamma_val * dy_val;
  }
  __syncthreads();
  for (I elem_id = tid; elem_id < instance_size; elem_id += blockDim.x) {
    cuda::atomic::Add(gamma_diff + elem_id, gamma_diff_sum_buf[elem_id]);
    cuda::atomic::Add(beta_diff + elem_id, beta_diff_sum_buf[elem_id]);
  }
}

template<typename I>
__global__ void LayerNormParamGradHalfImpl(const I n, const I instance_size, const half* dy,
                                           const half* normalized, const half* gamma,
                                           half* tmp_gamma_diff, half* tmp_beta_diff,
                                           half* normalized_diff) {
  extern __shared__ __align__(sizeof(double)) unsigned char bw_shared_buf[];
  auto* gamma_diff_sum_buf = reinterpret_cast<float*>(bw_shared_buf);
  auto* beta_diff_sum_buf = gamma_diff_sum_buf + instance_size;
  const I tid = threadIdx.x;
  for (I elem_id = tid; elem_id < instance_size; elem_id += blockDim.x) {
    gamma_diff_sum_buf[elem_id] = 0;
    beta_diff_sum_buf[elem_id] = 0;
  }
  __syncthreads();
  CUDA_1D_KERNEL_LOOP_T(I, i, n) {
    const I elem_id = i % instance_size;
    half dy_val = dy[i];
    half normalized_val = normalized[i];
    cuda::atomic::Add(&gamma_diff_sum_buf[elem_id],
                      __half2float(dy_val) * __half2float(normalized_val));
    cuda::atomic::Add(&beta_diff_sum_buf[elem_id], __half2float(dy_val));
    half gamma_val = gamma[elem_id];
    normalized_diff[i] = __hmul(gamma_val, dy_val);
  }
  __syncthreads();
  for (I elem_id = tid; elem_id < instance_size; elem_id += blockDim.x) {
    const I offset = blockIdx.x * instance_size + elem_id;
    tmp_gamma_diff[offset] = __float2half(gamma_diff_sum_buf[elem_id]);
    tmp_beta_diff[offset] = __float2half(beta_diff_sum_buf[elem_id]);
  }
}

template<typename T, bool do_scale, bool do_center>
void LayerNormForwardGpu(ep::Stream* stream, const int64_t num_instances, const int64_t norm_size,
                         const double epsilon, const T* x_ptr, const T* gamma_ptr,
                         const T* beta_ptr, T* normalized_ptr, T* y_ptr, user_op::Tensor* mean,
                         user_op::Tensor* inv_variance) {
  using ComputeType = typename cuda::layer_norm::DefaultComputeType<T>::type;
  cuda::layer_norm::DirectLoad<T, ComputeType> load(x_ptr, norm_size);
  AffineStore<ComputeType, T, do_scale, do_center> store(normalized_ptr, y_ptr, norm_size,
                                                         gamma_ptr, beta_ptr);
  cuda::layer_norm::DispatchLayerNorm<decltype(load), decltype(store), ComputeType>(
      stream->As<ep::CudaStream>()->cuda_stream(), load, store, num_instances, norm_size, epsilon,
      mean->mut_dptr<ComputeType>(), inv_variance->mut_dptr<ComputeType>());
}

template<typename T>
void DispatchLayerNormForwardGpu(ep::Stream* stream, const int64_t num_instances,
                                 const int64_t norm_size, const double epsilon, const T* x_ptr,
                                 const T* gamma_ptr, const T* beta_ptr, T* normalized_ptr, T* y_ptr,
                                 user_op::Tensor* mean, user_op::Tensor* inv_variance) {
  if (gamma_ptr != nullptr && beta_ptr != nullptr) {
    LayerNormForwardGpu<T, true, true>(stream, num_instances, norm_size, epsilon, x_ptr, gamma_ptr,
                                       beta_ptr, normalized_ptr, y_ptr, mean, inv_variance);
  } else if (gamma_ptr != nullptr && beta_ptr == nullptr) {
    LayerNormForwardGpu<T, true, false>(stream, num_instances, norm_size, epsilon, x_ptr, gamma_ptr,
                                        beta_ptr, normalized_ptr, y_ptr, mean, inv_variance);
  } else if (gamma_ptr == nullptr && beta_ptr != nullptr) {
    LayerNormForwardGpu<T, false, true>(stream, num_instances, norm_size, epsilon, x_ptr, gamma_ptr,
                                        beta_ptr, normalized_ptr, y_ptr, mean, inv_variance);
  } else {
    LayerNormForwardGpu<T, false, false>(stream, num_instances, norm_size, epsilon, x_ptr,
                                         gamma_ptr, beta_ptr, normalized_ptr, y_ptr, mean,
                                         inv_variance);
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
    user_op::Tensor* normalized =
        ctx->has_input("gamma", 0) ? ctx->Tensor4ArgNameAndIndex("normalized", 0) : y;
    const double epsilon = ctx->Attr<double>("epsilon");
    CHECK_GE(epsilon, CUDNN_BN_MIN_EPSILON);
    const int64_t num_instances = mean->shape().elem_cnt();
    const int64_t norm_size = x->shape().elem_cnt() / num_instances;
    const T* gamma_ptr = nullptr;
    const T* beta_ptr = nullptr;
    if (ctx->has_input("gamma", 0)) {
      const user_op::Tensor* gamma = ctx->Tensor4ArgNameAndIndex("gamma", 0);
      gamma_ptr = gamma->dptr<T>();
      CHECK_EQ(gamma->shape().elem_cnt(), norm_size);
    }
    if (ctx->has_input("beta", 0)) { beta_ptr = ctx->Tensor4ArgNameAndIndex("beta", 0)->dptr<T>(); }
    DispatchLayerNormForwardGpu<T>(ctx->stream(), num_instances, norm_size, epsilon, x->dptr<T>(),
                                   gamma_ptr, beta_ptr, normalized->mut_dptr<T>(), y->mut_dptr<T>(),
                                   mean, inv_variance);
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

template<typename T, typename BNParamT>
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
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const size_t aligned_buffer_size =
        GetCudaAlignedSize(mean->shape().elem_cnt() * GetSizeOfDataType(mean->data_type()));
    char* cudnn_bn_scale_ones_dptr = tmp_buffer->mut_dptr<char>();
    char* cudnn_bn_scale_diff_buf_dptr = cudnn_bn_scale_ones_dptr + aligned_buffer_size;
    char* cudnn_bn_bias_diff_buf_dptr = cudnn_bn_scale_ones_dptr + aligned_buffer_size;
    auto fill = NewFillPrimitive(ctx->stream(), mean->data_type());
    fill->Launch(ctx->stream(), cudnn_bn_scale_ones_dptr, 1, mean->shape().elem_cnt());
    const void* sp_alpha = CudnnSPOnePtr<T>();
    const void* sp_beta = nullptr;
    if (ctx->has_input("_add_to_output", 0)) {
      const user_op::Tensor* add_to_output = ctx->Tensor4ArgNameAndIndex("_add_to_output", 0);
      CHECK_EQ(add_to_output->data_type(), dx->data_type());
      CHECK_EQ(add_to_output->shape(), dx->shape());
      Memcpy<DeviceType::kCUDA>(
          ctx->stream(), dx->mut_dptr<void>(), add_to_output->dptr<void>(),
          add_to_output->shape().elem_cnt() * GetSizeOfDataType(add_to_output->data_type()));
      sp_beta = CudnnSPOnePtr<T>();
    } else {
      sp_beta = CudnnSPZeroPtr<T>();
    }
    const double epsilon = ctx->Attr<double>("epsilon");
    CHECK_GE(epsilon, CUDNN_BN_MIN_EPSILON);
    LayerNormCudnnBnCtx bn_ctx(x->shape(), mean->shape(), x->data_type());
    OF_CUDNN_CHECK(cudnnBatchNormalizationBackward(
        ctx->stream()->As<ep::CudaStream>()->cudnn_handle(), bn_ctx.mode(), sp_alpha, sp_beta,
        CudnnSPOnePtr<T>(), CudnnSPZeroPtr<T>(), bn_ctx.data_tensor_desc(), x->dptr<T>(),
        bn_ctx.data_tensor_desc(), dy->dptr<T>(), bn_ctx.data_tensor_desc(), dx->mut_dptr<T>(),
        bn_ctx.param_tensor_desc(), reinterpret_cast<const BNParamT*>(cudnn_bn_scale_ones_dptr),
        reinterpret_cast<BNParamT*>(cudnn_bn_scale_diff_buf_dptr),
        reinterpret_cast<BNParamT*>(cudnn_bn_bias_diff_buf_dptr), epsilon, mean->dptr(),
        inv_variance->dptr()));
  };
};

#define REGISTER_LAYER_NORM_GRAD_CUDA_KERNEL(dtype, bn_param_dtype)                        \
  REGISTER_USER_KERNEL("layer_norm_grad")                                                  \
      .SetCreateFn<LayerNormGradGpuKernel<dtype, bn_param_dtype>>()                        \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                     \
                       && (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value))    \
      .SetInferTmpSizeFn([](oneflow::user_op::InferContext* ctx) {                         \
        const user_op::TensorDesc& mean = ctx->InputTensorDesc("mean", 0);                 \
        const DataType& data_type = mean.data_type();                                      \
        const int64_t elem_cnt = mean.shape().elem_cnt();                                  \
        return GetCudaAlignedSize(elem_cnt * GetSizeOfDataType(data_type)) * 3;            \
      })                                                                                   \
      .SetInplaceProposalFn(                                                               \
          [](const user_op::InferContext& ctx,                                             \
             const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {       \
            if (ctx.has_input("_add_to_output", 0)) {                                      \
              OF_RETURN_IF_ERROR(AddInplaceArgPairFn("dx", 0, "_add_to_output", 0, true)); \
            }                                                                              \
            return Maybe<void>::Ok();                                                      \
          });

REGISTER_LAYER_NORM_GRAD_CUDA_KERNEL(float, float)
REGISTER_LAYER_NORM_GRAD_CUDA_KERNEL(double, double)
REGISTER_LAYER_NORM_GRAD_CUDA_KERNEL(float16, float)

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
    using NdUtil = NdarrayUtil<DeviceType::kCUDA, T>;
    auto Val = NdUtil::GetValNdarrayBuilder();
    auto Var = NdUtil::GetVarNdarrayBuilder();
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* beta_diff = ctx->Tensor4ArgNameAndIndex("beta_diff", 0);
    user_op::Tensor* gamma_diff = ctx->Tensor4ArgNameAndIndex("gamma_diff", 0);
    user_op::Tensor* normalized_diff = ctx->Tensor4ArgNameAndIndex("normalized_diff", 0);
    user_op::Tensor* gamma = ctx->Tensor4ArgNameAndIndex("gamma", 0);
    const bool has_beta_diff = beta_diff != nullptr;
    const bool has_gamma_diff = gamma_diff != nullptr;
    const bool has_normalized_diff = normalized_diff != nullptr;
    const bool has_gamma = gamma != nullptr;
    const int64_t begin_params_axis = ctx->Attr<int64_t>("begin_params_axis");
    const int64_t elem_cnt = dy->shape().elem_cnt();
    const int64_t m = dy->shape().Count(begin_params_axis);
    int max_active_blocks = 0;
    OF_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks, LayerNormParamGradImpl<T, int64_t>, GetLayerNormParamGradBlockSize(),
        GetParamGradDynamicSharedMemorySize<T>(m)));
    if (has_gamma_diff && has_beta_diff && has_normalized_diff && max_active_blocks > 0) {
      const user_op::Tensor* normalized = ctx->Tensor4ArgNameAndIndex("normalized", 0);
      Memset<DeviceType::kCUDA>(ctx->stream(), gamma_diff->mut_dptr<T>(), 0,
                                gamma_diff->shape().elem_cnt() * sizeof(T));
      Memset<DeviceType::kCUDA>(ctx->stream(), beta_diff->mut_dptr<T>(), 0,
                                beta_diff->shape().elem_cnt() * sizeof(T));
      if (elem_cnt > static_cast<int64_t>(GetMaxVal<int32_t>() / 2)) {
        LayerNormParamGradImpl<T, int64_t>
            <<<GetLayerNormParamGradNumBlocks(elem_cnt), GetLayerNormParamGradBlockSize(),
               GetParamGradDynamicSharedMemorySize<T>(m),
               ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
                elem_cnt, m, dy->dptr<T>(), normalized->dptr<T>(), gamma->dptr<T>(),
                gamma_diff->mut_dptr<T>(), beta_diff->mut_dptr<T>(),
                normalized_diff->mut_dptr<T>());
      } else {
        LayerNormParamGradImpl<T, int32_t>
            <<<GetLayerNormParamGradNumBlocks(elem_cnt), GetLayerNormParamGradBlockSize(),
               GetParamGradDynamicSharedMemorySize<T>(m),
               ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
                static_cast<int32_t>(elem_cnt), static_cast<int32_t>(m), dy->dptr<T>(),
                normalized->dptr<T>(), gamma->dptr<T>(), gamma_diff->mut_dptr<T>(),
                beta_diff->mut_dptr<T>(), normalized_diff->mut_dptr<T>());
      }
    } else {
      if (has_beta_diff) {
        user_op::Tensor* reduce_buf = ctx->Tensor4ArgNameAndIndex("reduce_buf", 0);
        CHECK_EQ(m, beta_diff->shape().elem_cnt());
        CHECK_EQ(dy->shape().elem_cnt() % m, 0);
        const int64_t n = dy->shape().elem_cnt() / m;
        NdUtil::ReduceSum(ctx->stream(), Var({1, m}, beta_diff->mut_dptr<T>()),
                          Val({n, m}, dy->dptr<T>()), Var({n, m}, reduce_buf->mut_dptr<T>()));
      }
      if (has_gamma_diff) {
        const user_op::Tensor* normalized = ctx->Tensor4ArgNameAndIndex("normalized", 0);
        user_op::Tensor* reduce_buf = ctx->Tensor4ArgNameAndIndex("reduce_buf", 0);
        CHECK_EQ(m, gamma_diff->shape().elem_cnt());
        CHECK_EQ(dy->shape().elem_cnt() % m, 0);
        const int64_t n = dy->shape().elem_cnt() / m;
        NdUtil::BroadcastMul(ctx->stream(), Var({n, m}, reduce_buf->mut_dptr<T>()),
                             Val({n, m}, normalized->dptr<T>()), Val({n, m}, dy->dptr<T>()));
        NdUtil::ReduceSum(ctx->stream(), Var({1, m}, gamma_diff->mut_dptr<T>()),
                          Val({n, m}, reduce_buf->dptr<T>()),
                          Var({n, m}, reduce_buf->mut_dptr<T>()));
      }
      if (has_normalized_diff) {
        if (has_gamma) {
          CHECK_EQ(m, gamma->shape().elem_cnt());
          CHECK_EQ(dy->shape().elem_cnt() % m, 0);
          const int64_t n = dy->shape().elem_cnt() / m;
          NdUtil::BroadcastMul(ctx->stream(), Var({n, m}, normalized_diff->mut_dptr<T>()),
                               Val({n, m}, dy->dptr<T>()), Val({1, m}, gamma->dptr<T>()));
        } else {
          Memcpy<DeviceType::kCUDA>(ctx->stream(), normalized_diff->mut_dptr<void>(),
                                    dy->dptr<void>(),
                                    dy->shape().elem_cnt() * GetSizeOfDataType(dy->data_type()));
        }
      }
    }
  };
};

#define REGISTER_LAYER_NORM_PARAM_GRAD_CUDA_KERNEL(dtype)              \
  REGISTER_USER_KERNEL("layer_norm_param_grad")                        \
      .SetCreateFn<LayerNormParamGradGpuKernel<dtype>>()               \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value));

REGISTER_LAYER_NORM_PARAM_GRAD_CUDA_KERNEL(float)
REGISTER_LAYER_NORM_PARAM_GRAD_CUDA_KERNEL(double)

class LayerNormParamGradGpuHalfKernel final : public user_op::OpKernel,
                                              public user_op::CudaGraphSupport {
 public:
  LayerNormParamGradGpuHalfKernel() = default;
  ~LayerNormParamGradGpuHalfKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx) const override {
    using NdUtil = NdarrayUtil<DeviceType::kCUDA, float16>;
    auto Val = NdUtil::GetValNdarrayBuilder();
    auto Var = NdUtil::GetVarNdarrayBuilder();
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* beta_diff = ctx->Tensor4ArgNameAndIndex("beta_diff", 0);
    user_op::Tensor* gamma_diff = ctx->Tensor4ArgNameAndIndex("gamma_diff", 0);
    user_op::Tensor* normalized_diff = ctx->Tensor4ArgNameAndIndex("normalized_diff", 0);
    user_op::Tensor* gamma = ctx->Tensor4ArgNameAndIndex("gamma", 0);
    const bool has_beta_diff = beta_diff != nullptr;
    const bool has_gamma_diff = gamma_diff != nullptr;
    const bool has_normalized_diff = normalized_diff != nullptr;
    const bool has_gamma = gamma != nullptr;
    const int64_t begin_params_axis = ctx->Attr<int64_t>("begin_params_axis");
    const int64_t elem_cnt = dy->shape().elem_cnt();
    const int64_t m = dy->shape().Count(begin_params_axis);
    int max_active_blocks = 0;
    OF_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks, LayerNormParamGradHalfImpl<int64_t>, GetLayerNormParamGradBlockSize(),
        GetParamGradDynamicSharedMemorySize<float16>(m)));
    if (has_gamma_diff && has_beta_diff && has_normalized_diff && max_active_blocks > 0) {
      const user_op::Tensor* normalized = ctx->Tensor4ArgNameAndIndex("normalized", 0);
      user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
      const int64_t num_blocks = GetLayerNormParamGradNumBlocks(dy->shape().elem_cnt());
      const size_t tmp_diff_size = GetCudaAlignedSize(num_blocks * m * sizeof(float16));
      float16* tmp_gamma_diff = tmp_buffer->mut_dptr<float16>();
      float16* tmp_beta_diff =
          reinterpret_cast<float16*>(tmp_buffer->mut_dptr<char>() + tmp_diff_size);
      float16* tmp_reduce_buf =
          reinterpret_cast<float16*>(tmp_buffer->mut_dptr<char>() + 2 * tmp_diff_size);
      CHECK_GE(tmp_buffer->shape().elem_cnt(), 3 * tmp_diff_size);
      if (elem_cnt > static_cast<int64_t>(GetMaxVal<int32_t>() / 2)) {
        LayerNormParamGradHalfImpl<int64_t>
            <<<GetLayerNormParamGradNumBlocks(elem_cnt), GetLayerNormParamGradBlockSize(),
               GetParamGradDynamicSharedMemorySize<float16>(m),
               ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
                elem_cnt, m, dy->dptr<half>(), normalized->dptr<half>(), gamma->dptr<half>(),
                reinterpret_cast<half*>(tmp_gamma_diff), reinterpret_cast<half*>(tmp_beta_diff),
                normalized_diff->mut_dptr<half>());
      } else {
        LayerNormParamGradHalfImpl<int32_t>
            <<<GetLayerNormParamGradNumBlocks(elem_cnt), GetLayerNormParamGradBlockSize(),
               GetParamGradDynamicSharedMemorySize<float16>(m),
               ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
                static_cast<int32_t>(elem_cnt), static_cast<int32_t>(m), dy->dptr<half>(),
                normalized->dptr<half>(), gamma->dptr<half>(),
                reinterpret_cast<half*>(tmp_gamma_diff), reinterpret_cast<half*>(tmp_beta_diff),
                normalized_diff->mut_dptr<half>());
      }
      NdUtil::ReduceSum(ctx->stream(), Var({1, m}, gamma_diff->mut_dptr<float16>()),
                        Val({num_blocks, m}, tmp_gamma_diff), Var({num_blocks, m}, tmp_reduce_buf));
      NdUtil::ReduceSum(ctx->stream(), Var({1, m}, beta_diff->mut_dptr<float16>()),
                        Val({num_blocks, m}, tmp_beta_diff), Var({num_blocks, m}, tmp_reduce_buf));
    } else {
      if (has_beta_diff) {
        user_op::Tensor* reduce_buf = ctx->Tensor4ArgNameAndIndex("reduce_buf", 0);
        CHECK_EQ(m, beta_diff->shape().elem_cnt());
        CHECK_EQ(dy->shape().elem_cnt() % m, 0);
        const int64_t n = dy->shape().elem_cnt() / m;
        NdUtil::ReduceSum(ctx->stream(), Var({1, m}, beta_diff->mut_dptr<float16>()),
                          Val({n, m}, dy->dptr<float16>()),
                          Var({n, m}, reduce_buf->mut_dptr<float16>()));
      }
      if (has_gamma_diff) {
        const user_op::Tensor* normalized = ctx->Tensor4ArgNameAndIndex("normalized", 0);
        user_op::Tensor* reduce_buf = ctx->Tensor4ArgNameAndIndex("reduce_buf", 0);
        CHECK_EQ(m, gamma_diff->shape().elem_cnt());
        CHECK_EQ(dy->shape().elem_cnt() % m, 0);
        const int64_t n = dy->shape().elem_cnt() / m;
        NdUtil::BroadcastMul(ctx->stream(), Var({n, m}, reduce_buf->mut_dptr<float16>()),
                             Val({n, m}, normalized->dptr<float16>()),
                             Val({n, m}, dy->dptr<float16>()));
        NdUtil::ReduceSum(ctx->stream(), Var({1, m}, gamma_diff->mut_dptr<float16>()),
                          Val({n, m}, reduce_buf->dptr<float16>()),
                          Var({n, m}, reduce_buf->mut_dptr<float16>()));
      }
      if (has_normalized_diff) {
        if (has_gamma) {
          CHECK_EQ(m, gamma->shape().elem_cnt());
          CHECK_EQ(dy->shape().elem_cnt() % m, 0);
          const int64_t n = dy->shape().elem_cnt() / m;
          NdUtil::BroadcastMul(ctx->stream(), Var({n, m}, normalized_diff->mut_dptr<float16>()),
                               Val({n, m}, dy->dptr<float16>()),
                               Val({1, m}, gamma->dptr<float16>()));
        } else {
          Memcpy<DeviceType::kCUDA>(ctx->stream(), normalized_diff->mut_dptr<void>(),
                                    dy->dptr<void>(),
                                    dy->shape().elem_cnt() * GetSizeOfDataType(dy->data_type()));
        }
      }
    }
  }
};

REGISTER_USER_KERNEL("layer_norm_param_grad")
    .SetCreateFn<LayerNormParamGradGpuHalfKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)
                     && (user_op::HobDataType("dy", 0) == DataType::kFloat16))
    .SetInferTmpSizeFn([](user_op::InferContext* ctx) {
      const int64_t begin_params_axis = ctx->Attr<int64_t>("begin_params_axis");
      const bool has_gamma_diff = ctx->has_output("gamma_diff", 0);
      const bool has_beta_diff = ctx->has_output("beta_diff", 0);
      const bool has_normalized_diff = ctx->has_output("normalized_diff", 0);
      const auto& dy = ctx->InputTensorDesc("dy", 0);
      const int64_t instance_size = dy.shape().Count(begin_params_axis);
      size_t tmp_buffer_size = 0;
      if (has_gamma_diff && has_beta_diff && has_normalized_diff) {
        const size_t tmp_gamma_diff =
            GetCudaAlignedSize(GetLayerNormParamGradNumBlocks(dy.shape().elem_cnt()) * instance_size
                               * sizeof(float16));
        const size_t tmp_beta_diff = tmp_gamma_diff;
        const size_t tmp_reduce_buf = tmp_gamma_diff;
        tmp_buffer_size = tmp_gamma_diff + tmp_beta_diff + tmp_reduce_buf;
      } else {
        tmp_buffer_size = 0;
      }
      return tmp_buffer_size;
    });
}  // namespace oneflow
