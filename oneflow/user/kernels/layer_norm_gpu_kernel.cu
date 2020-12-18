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
#include "oneflow/core/kernel/kernel_util.cuh"
#include <cub/cub.cuh>

namespace oneflow {

namespace {

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

template<typename T, bool do_scale, bool do_center>
__global__ void InstanceScaleCenterGpu(const int64_t elem_cnt, const int64_t instance_size,
                                       const T* in, const T* gamma, const T* beta, T* out) {
  CUDA_1D_KERNEL_LOOP_T(int64_t, i, elem_cnt) {
    const int64_t elem_id = i % instance_size;
    T v = in[i];
    if (do_scale) { v *= gamma[elem_id]; }
    if (do_center) { v += beta[elem_id]; }
    out[i] = v;
  }
}

template<bool do_scale, bool do_center>
__global__ void InstanceScaleCenterH2Gpu(const int64_t h2_elem_cnt, const int64_t h2_instance_size,
                                         const half* in, const half* gamma, const half* beta,
                                         half* out) {
  const auto* in_h2 = reinterpret_cast<const half2*>(in);
  const auto* gamma_h2 = reinterpret_cast<const half2*>(gamma);
  const auto* beta_h2 = reinterpret_cast<const half2*>(beta);
  auto* out_h2 = reinterpret_cast<half2*>(out);
  CUDA_1D_KERNEL_LOOP_T(int64_t, i, h2_elem_cnt) {
    const int64_t elem_id = i % h2_instance_size;
    half2 v2 = in_h2[i];
    if (do_scale) { v2 = __hmul2(v2, gamma_h2[elem_id]); }
    if (do_center) { v2 = __hadd2(v2, beta_h2[elem_id]); }
    out_h2[i] = v2;
  }
}

template<typename T>
void InstanceScaleCenter(DeviceCtx* ctx, const int64_t batch_size, const int64_t instance_size,
                         const T* in, const T* gamma, const T* beta, T* out) {
  const int64_t elem_cnt = batch_size * instance_size;
  if (beta != nullptr && gamma != nullptr) {  // scale and center
    InstanceScaleCenterGpu<T, true, true>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            elem_cnt, instance_size, in, gamma, beta, out);
  } else if (gamma != nullptr) {  // scale only
    InstanceScaleCenterGpu<T, true, false>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            elem_cnt, instance_size, in, gamma, nullptr, out);
  } else if (beta != nullptr) {  // center only
    InstanceScaleCenterGpu<T, false, true>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            elem_cnt, instance_size, in, nullptr, beta, out);
  } else {
    UNIMPLEMENTED();
  }
}

void InstanceScaleCenterH2(DeviceCtx* ctx, const int64_t batch_size, const int64_t instance_size,
                           const half* in, const half* gamma, const half* beta, half* out) {
  CHECK_EQ(instance_size % 2, 0);
  const int64_t elem_cnt_h2 = batch_size * instance_size / 2;
  const int64_t instance_size_h2 = instance_size / 2;
  if (beta != nullptr && gamma != nullptr) {  // scale and center
    InstanceScaleCenterH2Gpu<true, true>
        <<<BlocksNum4ThreadsNum(elem_cnt_h2), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            elem_cnt_h2, instance_size_h2, in, gamma, beta, out);
  } else if (gamma != nullptr) {  // scale only
    InstanceScaleCenterH2Gpu<true, false>
        <<<BlocksNum4ThreadsNum(elem_cnt_h2), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            elem_cnt_h2, instance_size_h2, in, gamma, nullptr, out);
  } else if (beta != nullptr) {  // center only
    InstanceScaleCenterH2Gpu<false, true>
        <<<BlocksNum4ThreadsNum(elem_cnt_h2), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            elem_cnt_h2, instance_size_h2, in, nullptr, beta, out);
  } else {
    UNIMPLEMENTED();
  }
}

template<>
void InstanceScaleCenter<float16>(DeviceCtx* ctx, const int64_t batch_size,
                                  const int64_t instance_size, const float16* in,
                                  const float16* gamma, const float16* beta, float16* out) {
  if (instance_size % 2 == 0) {
    InstanceScaleCenterH2(ctx, batch_size, instance_size, reinterpret_cast<const half*>(in),
                          reinterpret_cast<const half*>(gamma), reinterpret_cast<const half*>(beta),
                          reinterpret_cast<half*>(out));
  } else {
    InstanceScaleCenter<half>(ctx, batch_size, instance_size, reinterpret_cast<const half*>(in),
                              reinterpret_cast<const half*>(gamma),
                              reinterpret_cast<const half*>(beta), reinterpret_cast<half*>(out));
  }
}

constexpr int64_t kLayerNormForwardGpuBlockSize = 256;

template<typename T>
struct LayerNormUtil {
  using ComputeType = T;
  __device__ static ComputeType ToComputeType(T v) { return v; }
  __device__ static T FromComputeType(ComputeType v) { return v; }
};

template<>
struct LayerNormUtil<half> {
  using ComputeType = float;
  __device__ static ComputeType ToComputeType(half v) { return __half2float(v); }
  __device__ static half FromComputeType(ComputeType v) { return __float2half(v); }
};

template<typename T>
int GetForwardDynamicSharedMemorySize(const int norm_size) {
  return norm_size * sizeof(typename LayerNormUtil<T>::ComputeType);
}

int GetLayerNormForwardBlockSize() { return kLayerNormForwardGpuBlockSize; }

int GetLayerNormForwardNumBlocks(const int num_instances) {
  return std::min(static_cast<int>(num_instances), kCudaMaxBlocksNum);
}

template<typename T, typename ComputeType>
__global__ void LayerNormForwardImpl(const int num_instances, const int norm_size,
                                     const double epsilon, const T* x, const T* gamma,
                                     const T* beta, ComputeType* mean, ComputeType* inv_variance,
                                     T* normalized, T* y) {
  using LU = LayerNormUtil<T>;
  extern __shared__ __align__(sizeof(ComputeType)) unsigned char fw_shared_buf[];
  auto* compute_buf = reinterpret_cast<ComputeType*>(fw_shared_buf);
  __shared__ ComputeType row_mean_shared;
  __shared__ ComputeType row_inv_var_shared;
  typedef cub::BlockReduce<ComputeType, kLayerNormForwardGpuBlockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage cub_mean_reduce_tmp_storage;
  __shared__ typename BlockReduce::TempStorage cub_variance_reduce_tmp_storage;
  ComputeType inv_norm_size = static_cast<ComputeType>(1.0) / static_cast<ComputeType>(norm_size);
  for (int row = blockIdx.x; row < num_instances; row += gridDim.x) {
    const int row_offset = row * norm_size;
    const T* x_row = x + row_offset;
    ComputeType thread_sum = 0;
    ComputeType thread_square_sum = 0;
    const int tid = threadIdx.x;
    for (int col = tid; col < norm_size; col += blockDim.x) {
      const ComputeType val = LU::ToComputeType(x_row[col]);
      compute_buf[col] = val;
      thread_sum += val;
      thread_square_sum += val * val;
    }
    __syncthreads();
    ComputeType block_sum = BlockReduce(cub_mean_reduce_tmp_storage).Reduce(thread_sum, cub::Sum());
    ComputeType block_square_sum =
        BlockReduce(cub_variance_reduce_tmp_storage).Reduce(thread_square_sum, cub::Sum());
    if (tid == 0) {
      ComputeType row_mean = block_sum * inv_norm_size;
      row_mean_shared = row_mean;
      mean[row] = row_mean;
      ComputeType row_variance =
          max(block_square_sum * inv_norm_size - row_mean * row_mean, static_cast<ComputeType>(0));
      ComputeType row_inv_var = rsqrt(row_variance + static_cast<ComputeType>(epsilon));
      row_inv_var_shared = row_inv_var;
      inv_variance[row] = row_inv_var;
    }
    __syncthreads();
    ComputeType mean = row_mean_shared;
    ComputeType inv_var = row_inv_var_shared;
    for (int col = threadIdx.x; col < norm_size; col += blockDim.x) {
      int offset = row_offset + col;
      ComputeType val = compute_buf[col];
      val = (val - mean) * inv_var;
      if (gamma != nullptr || beta != nullptr) {
        int elem_id = col;
        if (gamma != nullptr) {
          normalized[offset] = LU::FromComputeType(val);
          val *= LU::ToComputeType(gamma[elem_id]);
        }
        if (beta != nullptr) { val += LU::ToComputeType(beta[elem_id]); }
      }
      y[offset] = LU::FromComputeType(val);
    }
  }
}

template<typename T>
void LayerNormForwardGpu(DeviceCtx* ctx, const int num_instances, const int norm_size,
                         const double epsilon, const T* x_ptr, const T* gamma_ptr,
                         const T* beta_ptr, T* normalized_ptr, T* y_ptr, user_op::Tensor* mean,
                         user_op::Tensor* inv_variance) {
  LayerNormForwardImpl<T, typename LayerNormUtil<T>::ComputeType>
      <<<GetLayerNormForwardNumBlocks(num_instances), GetLayerNormForwardBlockSize(),
         GetForwardDynamicSharedMemorySize<T>(norm_size), ctx->cuda_stream()>>>(
          num_instances, norm_size, epsilon, x_ptr, gamma_ptr, beta_ptr,
          mean->mut_dptr<typename LayerNormUtil<T>::ComputeType>(),
          inv_variance->mut_dptr<typename LayerNormUtil<T>::ComputeType>(), normalized_ptr, y_ptr);
}

template<>
void LayerNormForwardGpu<float16>(DeviceCtx* ctx, const int num_instances, const int norm_size,
                                  const double epsilon, const float16* x_ptr,
                                  const float16* gamma_ptr, const float16* beta_ptr,
                                  float16* normalized_ptr, float16* y_ptr, user_op::Tensor* mean,
                                  user_op::Tensor* inv_variance) {
  LayerNormForwardImpl<half, typename LayerNormUtil<half>::ComputeType>
      <<<GetLayerNormForwardNumBlocks(num_instances), GetLayerNormForwardBlockSize(),
         GetForwardDynamicSharedMemorySize<half>(norm_size), ctx->cuda_stream()>>>(
          num_instances, norm_size, epsilon, reinterpret_cast<const half*>(x_ptr),
          reinterpret_cast<const half*>(gamma_ptr), reinterpret_cast<const half*>(beta_ptr),
          mean->mut_dptr<typename LayerNormUtil<half>::ComputeType>(),
          inv_variance->mut_dptr<typename LayerNormUtil<half>::ComputeType>(),
          reinterpret_cast<half*>(normalized_ptr), reinterpret_cast<half*>(y_ptr));
}

int GetForwardFusedKernelMinNormSize() { return 64; }

template<typename T>
int GetForwardFusedKernelMaxActiveBlocks(const int32_t norm_size) {
  int max_active_blocks;
  OF_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_blocks, LayerNormForwardImpl<T, typename LayerNormUtil<T>::ComputeType>,
      GetLayerNormForwardBlockSize(), GetForwardDynamicSharedMemorySize<T>(norm_size)));
  return max_active_blocks;
}

template<>
int GetForwardFusedKernelMaxActiveBlocks<float16>(const int32_t norm_size) {
  int max_active_blocks;
  OF_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_blocks, LayerNormForwardImpl<half, typename LayerNormUtil<half>::ComputeType>,
      GetLayerNormForwardBlockSize(), GetForwardDynamicSharedMemorySize<half>(norm_size)));
  return max_active_blocks;
}

template<typename T>
bool IsForwardFusedKernelSupported(const int32_t norm_size, const int32_t instance_size) {
  if (norm_size >= GetForwardFusedKernelMinNormSize() && norm_size % 32 == 0
      && GetForwardFusedKernelMaxActiveBlocks<T>(norm_size) > 0
      && (instance_size == 0 || norm_size == instance_size)) {
    return true;
  } else {
    return false;
  }
}

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
  extern __shared__ __align__(sizeof(T)) unsigned char bw_shared_buf[];
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
    gpu_atomic_add(&gamma_diff_sum_buf[elem_id], dy_val * normalized_val);
    gpu_atomic_add(&beta_diff_sum_buf[elem_id], dy_val);
    T gamma_val = gamma[elem_id];
    normalized_diff[i] = gamma_val * dy_val;
  }
  __syncthreads();
  for (I elem_id = tid; elem_id < instance_size; elem_id += blockDim.x) {
    gpu_atomic_add(gamma_diff + elem_id, gamma_diff_sum_buf[elem_id]);
    gpu_atomic_add(beta_diff + elem_id, beta_diff_sum_buf[elem_id]);
  }
}

template<typename I>
__global__ void LayerNormParamGradHalfImpl(const I n, const I instance_size, const half* dy,
                                           const half* normalized, const half* gamma,
                                           half* tmp_gamma_diff, half* tmp_beta_diff,
                                           half* normalized_diff) {
  extern __shared__ __align__(sizeof(float)) unsigned char bw_shared_buf[];
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
    gpu_atomic_add(&gamma_diff_sum_buf[elem_id],
                   __half2float(dy_val) * __half2float(normalized_val));
    gpu_atomic_add(&beta_diff_sum_buf[elem_id], __half2float(dy_val));
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

}  // namespace

template<typename T, typename BNParamT>
class LayerNormGpuKernel final : public user_op::OpKernel {
 public:
  LayerNormGpuKernel() = default;
  ~LayerNormGpuKernel() = default;

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* mean = ctx->Tensor4ArgNameAndIndex("mean", 0);
    user_op::Tensor* inv_variance = ctx->Tensor4ArgNameAndIndex("inv_variance", 0);
    const bool scale = ctx->Attr<bool>("scale");
    const bool center = ctx->Attr<bool>("center");
    user_op::Tensor* normalized = scale ? ctx->Tensor4ArgNameAndIndex("normalized", 0) : y;
    const double epsilon = ctx->Attr<double>("epsilon");
    CHECK_GE(epsilon, CUDNN_BN_MIN_EPSILON);
    const int32_t num_instances = mean->shape().elem_cnt();
    const int32_t norm_size = x->shape().elem_cnt() / num_instances;
    int32_t instance_size = 0;
    const T* gamma_ptr = nullptr;
    const T* beta_ptr = nullptr;
    if (scale || center) {
      if (scale) {
        const user_op::Tensor* gamma = ctx->Tensor4ArgNameAndIndex("gamma", 0);
        instance_size = gamma->shape().elem_cnt();
        gamma_ptr = gamma->dptr<T>();
      }
      if (center) {
        const user_op::Tensor* beta = ctx->Tensor4ArgNameAndIndex("beta", 0);
        if (gamma_ptr) {
          CHECK_EQ(beta->shape().elem_cnt(), instance_size);
        } else {
          instance_size = beta->shape().elem_cnt();
        }
        beta_ptr = beta->dptr<T>();
      }
      CHECK_EQ(y->shape().elem_cnt() % instance_size, 0);
    }
    if (IsForwardFusedKernelSupported<T>(norm_size, instance_size)) {
      LayerNormForwardGpu<T>(ctx->device_ctx(), num_instances, norm_size, epsilon, x->dptr<T>(),
                             gamma_ptr, beta_ptr, normalized->mut_dptr<T>(), y->mut_dptr<T>(), mean,
                             inv_variance);
    } else {
      LayerNormCudnnBnCtx bn_ctx(x->shape(), mean->shape(), x->data_type());
      user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
      const size_t aligned_buffer_size =
          GetCudaAlignedSize(mean->shape().elem_cnt() * GetSizeOfDataType(mean->data_type()));
      char* cudnn_bn_scale_ones_dptr = tmp_buffer->mut_dptr<char>();
      char* cudnn_bn_bias_zeros_dptr = cudnn_bn_scale_ones_dptr + aligned_buffer_size;
      NewKernelUtil<DeviceType::kGPU>::Fill(ctx->device_ctx(), mean->shape().elem_cnt(),
                                            static_cast<BNParamT>(1),
                                            reinterpret_cast<BNParamT*>(cudnn_bn_scale_ones_dptr));
      NewKernelUtil<DeviceType::kGPU>::Fill(ctx->device_ctx(), mean->shape().elem_cnt(),
                                            static_cast<BNParamT>(0),
                                            reinterpret_cast<BNParamT*>(cudnn_bn_bias_zeros_dptr));
      OF_CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
          ctx->device_ctx()->cudnn_handle(), bn_ctx.mode(), CudnnSPOnePtr<T>(), CudnnSPZeroPtr<T>(),
          bn_ctx.data_tensor_desc(), x->dptr<T>(), bn_ctx.data_tensor_desc(),
          normalized->mut_dptr<T>(), bn_ctx.param_tensor_desc(),
          reinterpret_cast<BNParamT*>(cudnn_bn_scale_ones_dptr),
          reinterpret_cast<BNParamT*>(cudnn_bn_bias_zeros_dptr), 1.0, nullptr, nullptr, epsilon,
          mean->mut_dptr(), inv_variance->mut_dptr()));
      if (scale || center) {
        const int64_t batch_size = y->shape().elem_cnt() / instance_size;
        InstanceScaleCenter<T>(ctx->device_ctx(), batch_size, instance_size, normalized->dptr<T>(),
                               gamma_ptr, beta_ptr, y->mut_dptr<T>());
      }
    }
  };
};

#define REGISTER_LAYER_NORM_GPU_KERNEL(dtype, bn_param_dtype)                         \
  REGISTER_USER_KERNEL("layer_norm")                                                  \
      .SetCreateFn<LayerNormGpuKernel<dtype, bn_param_dtype>>()                       \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                             \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn([](oneflow::user_op::InferContext* ctx) {                    \
        user_op::TensorDesc* mean = ctx->TensorDesc4ArgNameAndIndex("mean", 0);       \
        const DataType& data_type = mean->data_type();                                \
        const int64_t elem_cnt = mean->shape().elem_cnt();                            \
        return GetCudaAlignedSize(elem_cnt * GetSizeOfDataType(data_type)) * 2;       \
      });

REGISTER_LAYER_NORM_GPU_KERNEL(float, float)
REGISTER_LAYER_NORM_GPU_KERNEL(double, double)
REGISTER_LAYER_NORM_GPU_KERNEL(float16, float)

template<typename T, typename BNParamT>
class LayerNormGradGpuKernel final : public user_op::OpKernel {
 public:
  LayerNormGradGpuKernel() = default;
  ~LayerNormGradGpuKernel() = default;

 private:
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
    NewKernelUtil<DeviceType::kGPU>::Fill(ctx->device_ctx(), mean->shape().elem_cnt(),
                                          static_cast<BNParamT>(1),
                                          reinterpret_cast<BNParamT*>(cudnn_bn_scale_ones_dptr));
    const void* sp_alpha = CudnnSPOnePtr<T>();
    const void* sp_beta;
    if (ctx->user_op_conf().has_input("_add_to_output", 0)) {
      const user_op::Tensor* add_to_output = ctx->Tensor4ArgNameAndIndex("_add_to_output", 0);
      CHECK_EQ(add_to_output->data_type(), dx->data_type());
      CHECK_EQ(add_to_output->shape(), dx->shape());
      Memcpy<DeviceType::kGPU>(
          ctx->device_ctx(), dx->mut_dptr<void>(), add_to_output->dptr<void>(),
          add_to_output->shape().elem_cnt() * GetSizeOfDataType(add_to_output->data_type()));
      sp_beta = CudnnSPOnePtr<T>();
    } else {
      sp_beta = CudnnSPZeroPtr<T>();
    }
    const double epsilon = ctx->Attr<double>("epsilon");
    CHECK_GE(epsilon, CUDNN_BN_MIN_EPSILON);
    LayerNormCudnnBnCtx bn_ctx(x->shape(), mean->shape(), x->data_type());
    OF_CUDNN_CHECK(cudnnBatchNormalizationBackward(
        ctx->device_ctx()->cudnn_handle(), bn_ctx.mode(), sp_alpha, sp_beta, CudnnSPOnePtr<T>(),
        CudnnSPZeroPtr<T>(), bn_ctx.data_tensor_desc(), x->dptr<T>(), bn_ctx.data_tensor_desc(),
        dy->dptr<T>(), bn_ctx.data_tensor_desc(), dx->mut_dptr<T>(), bn_ctx.param_tensor_desc(),
        reinterpret_cast<const BNParamT*>(cudnn_bn_scale_ones_dptr),
        reinterpret_cast<BNParamT*>(cudnn_bn_scale_diff_buf_dptr),
        reinterpret_cast<BNParamT*>(cudnn_bn_bias_diff_buf_dptr), epsilon, mean->dptr(),
        inv_variance->dptr()));
  };
};

#define REGISTER_LAYER_NORM_GRAD_GPU_KERNEL(dtype, bn_param_dtype)                              \
  REGISTER_USER_KERNEL("layer_norm_grad")                                                       \
      .SetCreateFn<LayerNormGradGpuKernel<dtype, bn_param_dtype>>()                             \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                                       \
                       & (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value))          \
      .SetInferTmpSizeFn([](oneflow::user_op::InferContext* ctx) {                              \
        user_op::TensorDesc* mean = ctx->TensorDesc4ArgNameAndIndex("mean", 0);                 \
        const DataType& data_type = mean->data_type();                                          \
        const int64_t elem_cnt = mean->shape().elem_cnt();                                      \
        return GetCudaAlignedSize(elem_cnt * GetSizeOfDataType(data_type)) * 3;                 \
      })                                                                                        \
      .SetInplaceProposalFn([](const user_op::InferContext& ctx,                                \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        if (ctx.user_op_conf().has_input("_add_to_output", 0)) {                                \
          OF_RETURN_IF_ERROR(AddInplaceArgPairFn("dx", 0, "_add_to_output", 0, true));          \
        }                                                                                       \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_LAYER_NORM_GRAD_GPU_KERNEL(float, float)
REGISTER_LAYER_NORM_GRAD_GPU_KERNEL(double, double)
REGISTER_LAYER_NORM_GRAD_GPU_KERNEL(float16, float)

template<typename T>
class LayerNormParamGradGpuKernel final : public user_op::OpKernel {
 public:
  LayerNormParamGradGpuKernel() = default;
  ~LayerNormParamGradGpuKernel() = default;

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx) const override {
    using NdUtil = NdarrayUtil<DeviceType::kGPU, T>;
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
    int max_active_blocks;
    OF_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks, LayerNormParamGradImpl<T, int64_t>, GetLayerNormParamGradBlockSize(),
        GetParamGradDynamicSharedMemorySize<T>(m)));
    if (has_gamma_diff && has_beta_diff && has_normalized_diff && max_active_blocks > 0) {
      const user_op::Tensor* normalized = ctx->Tensor4ArgNameAndIndex("normalized", 0);
      Memset<DeviceType::kGPU>(ctx->device_ctx(), gamma_diff->mut_dptr<T>(), 0,
                               gamma_diff->shape().elem_cnt() * sizeof(T));
      Memset<DeviceType::kGPU>(ctx->device_ctx(), beta_diff->mut_dptr<T>(), 0,
                               beta_diff->shape().elem_cnt() * sizeof(T));
      if (elem_cnt > static_cast<int64_t>(GetMaxVal<int32_t>() / 2)) {
        LayerNormParamGradImpl<T, int64_t>
            <<<GetLayerNormParamGradNumBlocks(elem_cnt), GetLayerNormParamGradBlockSize(),
               GetParamGradDynamicSharedMemorySize<T>(m), ctx->device_ctx()->cuda_stream()>>>(
                elem_cnt, m, dy->dptr<T>(), normalized->dptr<T>(), gamma->dptr<T>(),
                gamma_diff->mut_dptr<T>(), beta_diff->mut_dptr<T>(),
                normalized_diff->mut_dptr<T>());
      } else {
        LayerNormParamGradImpl<T, int32_t>
            <<<GetLayerNormParamGradNumBlocks(elem_cnt), GetLayerNormParamGradBlockSize(),
               GetParamGradDynamicSharedMemorySize<T>(m), ctx->device_ctx()->cuda_stream()>>>(
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
        NdUtil::ReduceSum(ctx->device_ctx(), Var({1, m}, beta_diff->mut_dptr<T>()),
                          Val({n, m}, dy->dptr<T>()), Var({n, m}, reduce_buf->mut_dptr<T>()));
      }
      if (has_gamma_diff) {
        const user_op::Tensor* normalized = ctx->Tensor4ArgNameAndIndex("normalized", 0);
        user_op::Tensor* reduce_buf = ctx->Tensor4ArgNameAndIndex("reduce_buf", 0);
        CHECK_EQ(m, gamma_diff->shape().elem_cnt());
        CHECK_EQ(dy->shape().elem_cnt() % m, 0);
        const int64_t n = dy->shape().elem_cnt() / m;
        NdUtil::BroadcastMul(ctx->device_ctx(), Var({n, m}, reduce_buf->mut_dptr<T>()),
                             Val({n, m}, normalized->dptr<T>()), Val({n, m}, dy->dptr<T>()));
        NdUtil::ReduceSum(ctx->device_ctx(), Var({1, m}, gamma_diff->mut_dptr<T>()),
                          Val({n, m}, reduce_buf->dptr<T>()),
                          Var({n, m}, reduce_buf->mut_dptr<T>()));
      }
      if (has_normalized_diff) {
        if (has_gamma) {
          CHECK_EQ(m, gamma->shape().elem_cnt());
          CHECK_EQ(dy->shape().elem_cnt() % m, 0);
          const int64_t n = dy->shape().elem_cnt() / m;
          NdUtil::BroadcastMul(ctx->device_ctx(), Var({n, m}, normalized_diff->mut_dptr<T>()),
                               Val({n, m}, dy->dptr<T>()), Val({1, m}, gamma->dptr<T>()));
        } else {
          Memcpy<DeviceType::kGPU>(ctx->device_ctx(), normalized_diff->mut_dptr<void>(),
                                   dy->dptr<void>(),
                                   dy->shape().elem_cnt() * GetSizeOfDataType(dy->data_type()));
        }
      }
    }
  };
};

#define REGISTER_LAYER_NORM_PARAM_GRAD_GPU_KERNEL(dtype)  \
  REGISTER_USER_KERNEL("layer_norm_param_grad")           \
      .SetCreateFn<LayerNormParamGradGpuKernel<dtype>>()  \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu") \
                       & (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value));

REGISTER_LAYER_NORM_PARAM_GRAD_GPU_KERNEL(float)
REGISTER_LAYER_NORM_PARAM_GRAD_GPU_KERNEL(double)

class LayerNormParamGradGpuHalfKernel final : public user_op::OpKernel {
 public:
  LayerNormParamGradGpuHalfKernel() = default;
  ~LayerNormParamGradGpuHalfKernel() = default;

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx) const override {
    using NdUtil = NdarrayUtil<DeviceType::kGPU, float16>;
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
    int max_active_blocks;
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
               GetParamGradDynamicSharedMemorySize<float16>(m), ctx->device_ctx()->cuda_stream()>>>(
                elem_cnt, m, dy->dptr<half>(), normalized->dptr<half>(), gamma->dptr<half>(),
                reinterpret_cast<half*>(tmp_gamma_diff), reinterpret_cast<half*>(tmp_beta_diff),
                normalized_diff->mut_dptr<half>());
      } else {
        LayerNormParamGradHalfImpl<int32_t>
            <<<GetLayerNormParamGradNumBlocks(elem_cnt), GetLayerNormParamGradBlockSize(),
               GetParamGradDynamicSharedMemorySize<float16>(m), ctx->device_ctx()->cuda_stream()>>>(
                static_cast<int32_t>(elem_cnt), static_cast<int32_t>(m), dy->dptr<half>(),
                normalized->dptr<half>(), gamma->dptr<half>(),
                reinterpret_cast<half*>(tmp_gamma_diff), reinterpret_cast<half*>(tmp_beta_diff),
                normalized_diff->mut_dptr<half>());
      }
      NdUtil::ReduceSum(ctx->device_ctx(), Var({1, m}, gamma_diff->mut_dptr<float16>()),
                        Val({num_blocks, m}, tmp_gamma_diff), Var({num_blocks, m}, tmp_reduce_buf));
      NdUtil::ReduceSum(ctx->device_ctx(), Var({1, m}, beta_diff->mut_dptr<float16>()),
                        Val({num_blocks, m}, tmp_beta_diff), Var({num_blocks, m}, tmp_reduce_buf));
    } else {
      if (has_beta_diff) {
        user_op::Tensor* reduce_buf = ctx->Tensor4ArgNameAndIndex("reduce_buf", 0);
        CHECK_EQ(m, beta_diff->shape().elem_cnt());
        CHECK_EQ(dy->shape().elem_cnt() % m, 0);
        const int64_t n = dy->shape().elem_cnt() / m;
        NdUtil::ReduceSum(ctx->device_ctx(), Var({1, m}, beta_diff->mut_dptr<float16>()),
                          Val({n, m}, dy->dptr<float16>()),
                          Var({n, m}, reduce_buf->mut_dptr<float16>()));
      }
      if (has_gamma_diff) {
        const user_op::Tensor* normalized = ctx->Tensor4ArgNameAndIndex("normalized", 0);
        user_op::Tensor* reduce_buf = ctx->Tensor4ArgNameAndIndex("reduce_buf", 0);
        CHECK_EQ(m, gamma_diff->shape().elem_cnt());
        CHECK_EQ(dy->shape().elem_cnt() % m, 0);
        const int64_t n = dy->shape().elem_cnt() / m;
        NdUtil::BroadcastMul(ctx->device_ctx(), Var({n, m}, reduce_buf->mut_dptr<float16>()),
                             Val({n, m}, normalized->dptr<float16>()),
                             Val({n, m}, dy->dptr<float16>()));
        NdUtil::ReduceSum(ctx->device_ctx(), Var({1, m}, gamma_diff->mut_dptr<float16>()),
                          Val({n, m}, reduce_buf->dptr<float16>()),
                          Var({n, m}, reduce_buf->mut_dptr<float16>()));
      }
      if (has_normalized_diff) {
        if (has_gamma) {
          CHECK_EQ(m, gamma->shape().elem_cnt());
          CHECK_EQ(dy->shape().elem_cnt() % m, 0);
          const int64_t n = dy->shape().elem_cnt() / m;
          NdUtil::BroadcastMul(ctx->device_ctx(), Var({n, m}, normalized_diff->mut_dptr<float16>()),
                               Val({n, m}, dy->dptr<float16>()),
                               Val({1, m}, gamma->dptr<float16>()));
        } else {
          Memcpy<DeviceType::kGPU>(ctx->device_ctx(), normalized_diff->mut_dptr<void>(),
                                   dy->dptr<void>(),
                                   dy->shape().elem_cnt() * GetSizeOfDataType(dy->data_type()));
        }
      }
    }
  }
};

REGISTER_USER_KERNEL("layer_norm_param_grad")
    .SetCreateFn<LayerNormParamGradGpuHalfKernel>()
    .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")
                     & (user_op::HobDataType("dy", 0) == DataType::kFloat16))
    .SetInferTmpSizeFn([](user_op::InferContext* ctx) {
      const int64_t begin_params_axis = ctx->Attr<int64_t>("begin_params_axis");
      const bool has_gamma_diff = ctx->user_op_conf().has_output("gamma_diff", 0);
      const bool has_beta_diff = ctx->user_op_conf().has_output("beta_diff", 0);
      const bool has_normalized_diff = ctx->user_op_conf().has_output("normalized_diff", 0);
      const auto* dy = ctx->TensorDesc4ArgNameAndIndex("dy", 0);
      const int64_t instance_size = dy->shape().Count(begin_params_axis);
      size_t tmp_buffer_size = 0;
      if (has_gamma_diff && has_beta_diff && has_normalized_diff) {
        const size_t tmp_gamma_diff =
            GetCudaAlignedSize(GetLayerNormParamGradNumBlocks(dy->shape().elem_cnt())
                               * instance_size * sizeof(float16));
        const size_t tmp_beta_diff = tmp_gamma_diff;
        const size_t tmp_reduce_buf = tmp_gamma_diff;
        tmp_buffer_size = tmp_gamma_diff + tmp_beta_diff + tmp_reduce_buf;
      } else {
        tmp_buffer_size = 0;
      }
      return tmp_buffer_size;
    });
}  // namespace oneflow
