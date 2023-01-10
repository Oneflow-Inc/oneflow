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
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/ep/cuda/primitive/unary_functor.cuh"
#include "oneflow/core/cuda/layer_norm.cuh"
#include <cub/cub.cuh>
#include "oneflow/core/kernel/cuda_graph_support.h"

#ifdef WITH_CUTLASS
#include <cutlass/fast_math.h>
#endif  // WITH_CUTLASS

namespace oneflow {

namespace {

template<typename SRC, typename DST, ep::primitive::UnaryOp activation, bool affine>
struct AffineStore {
  AffineStore(DST* y, int64_t row_size, int64_t channel_size, int64_t spatial_size,
              const DST* gamma, const DST* beta)
      : y(y),
        row_size(row_size),
        channel_size(channel_size),
        spatial_size(spatial_size),
        gamma(gamma),
        beta(beta),
        act(0, 0) {}

  template<int PackSize>
  __device__ void store(const SRC* src, int64_t row, int64_t col) {
    cuda::layer_norm::Pack<DST, PackSize> y_pack;
    const int64_t offset = row * row_size + col;
    const int64_t packed_offset = offset / PackSize;
    const int64_t gamma_beta_offset = (offset / spatial_size) % channel_size;
    DST gamma_val = 1.0;
    DST beta_val = 0.0;
    if (affine) {
      gamma_val = gamma[gamma_beta_offset];
      beta_val = beta[gamma_beta_offset];
    }

#pragma unroll
    for (int i = 0; i < PackSize; ++i) {
      DST normalized_i = static_cast<DST>(src[i]);
      if (affine) {
        y_pack.elem[i] = act(normalized_i * gamma_val + beta_val);
      } else {
        // Direct Store.
        y_pack.elem[i] = act(normalized_i);
      }
    }
    *(reinterpret_cast<cuda::layer_norm::PackType<DST, PackSize>*>(y) + packed_offset) =
        y_pack.storage;
  }
  bool CanPackAs(size_t pack_size) { return (spatial_size % pack_size) == 0; }
  DST* y;
  int64_t row_size;
  int64_t channel_size;
  int64_t spatial_size;
  const DST* gamma;
  const DST* beta;
  ep::primitive::UnaryFunctor<DeviceType::kCUDA, activation, DST, DST> act;
};

template<typename SRC, typename DST, bool affine>
struct ScaleLoad {
  using LoadType = DST;
  ScaleLoad(const SRC* src, const SRC* gamma, int64_t row_size, int64_t channel_size,
            int64_t spatial_size)
      : src(src),
        gamma(gamma),
        row_size(row_size),
        channel_size(channel_size),
        spatial_size(spatial_size) {}
  template<int PackSize>
  __device__ void load(DST* dst, int64_t row, int64_t col) const {
    cuda::layer_norm::Pack<SRC, PackSize> src_pack;
    cuda::layer_norm::Pack<SRC, PackSize> gamma_pack;

    const int64_t offset = row * row_size + col;
    const int64_t packed_offset = offset / PackSize;
    const int64_t gamma_offset = (offset / spatial_size) % channel_size;

    src_pack.storage =
        *(reinterpret_cast<const cuda::layer_norm::PackType<SRC, PackSize>*>(src) + packed_offset);
    SRC gamma_val = static_cast<SRC>(1.0);
    if (affine) { gamma_val = gamma[gamma_offset]; }
#pragma unroll
    for (int i = 0; i < PackSize; ++i) { dst[i] = static_cast<DST>(src_pack.elem[i] * gamma_val); }
  }
  bool CanPackAs(size_t pack_size) { return (spatial_size % pack_size) == 0; }
  const SRC* src;
  const SRC* gamma;
  int64_t row_size;
  int64_t channel_size;
  int64_t spatial_size;
};

#ifdef WITH_CUTLASS

template<typename SRC, typename DST, ep::primitive::UnaryOp activation, bool affine>
struct ChannelsLastStore {
  ChannelsLastStore(DST* y, const DST* gamma, const DST* beta, int64_t spatial_size,
                    int64_t channel_size, int64_t num_groups)
      : y(y),
        gamma(gamma),
        beta(beta),
        spatial_size(spatial_size),
        c0(num_groups),
        c1(channel_size / num_groups),
        act(0, 0) {}

  template<int PackSize>
  __device__ void store(const SRC* src, int32_t row, int32_t col) {
    cuda::layer_norm::Pack<DST, PackSize> y_pack;
    cuda::layer_norm::Pack<DST, PackSize> gamma_pack;
    cuda::layer_norm::Pack<DST, PackSize> beta_pack;
    int32_t spatial_idx;
    int32_t c1_idx;
    c1(spatial_idx, c1_idx, col);
    int32_t batch_idx;
    int32_t c0_idx;
    c0(batch_idx, c0_idx, row);
    const int32_t y_offset =
        (batch_idx * c0.divisor * c1.divisor * spatial_size + spatial_idx * c0.divisor * c1.divisor
         + c0_idx * c1.divisor + c1_idx)
        / PackSize;
    const int32_t gamma_beta_offset = (c0_idx * c1.divisor + c1_idx) / PackSize;
    if (affine) {
      gamma_pack.storage =
          *(reinterpret_cast<const cuda::layer_norm::PackType<DST, PackSize>*>(gamma)
            + gamma_beta_offset);
      beta_pack.storage = *(reinterpret_cast<const cuda::layer_norm::PackType<DST, PackSize>*>(beta)
                            + gamma_beta_offset);
    }

#pragma unroll
    for (int i = 0; i < PackSize; ++i) {
      DST normalized_i = static_cast<DST>(src[i]);
      if (affine) {
        y_pack.elem[i] = act(normalized_i * gamma_pack.elem[i] + beta_pack.elem[i]);
      } else {
        // Direct Store.
        y_pack.elem[i] = act(normalized_i);
      }
    }
    *(reinterpret_cast<cuda::layer_norm::PackType<DST, PackSize>*>(y) + y_offset) = y_pack.storage;
  }
  bool CanPackAs(size_t pack_size) { return (c1.divisor % pack_size) == 0; }
  DST* y;
  const DST* gamma;
  const DST* beta;
  int32_t spatial_size;
  cutlass::FastDivmod c0;
  cutlass::FastDivmod c1;
  ep::primitive::UnaryFunctor<DeviceType::kCUDA, activation, DST, DST> act;
};

template<typename SRC, typename DST>
struct ChannelsLastLoad {
  using LoadType = DST;
  ChannelsLastLoad(const SRC* src, int64_t spatial_size, int64_t channel_size, int64_t num_groups)
      : src(src), spatial_size(spatial_size), c0(num_groups), c1(channel_size / num_groups) {}
  template<int N>
  __device__ void load(DST* dst, int32_t row, int32_t col) const {
    int32_t spatial_idx;
    int32_t c1_idx;
    c1(spatial_idx, c1_idx, col);
    int32_t batch_idx;
    int32_t c0_idx;
    c0(batch_idx, c0_idx, row);
    cuda::layer_norm::Pack<SRC, N> pack;
    const int32_t offset = (batch_idx * c0.divisor * c1.divisor * spatial_size
                            + spatial_idx * c0.divisor * c1.divisor + c0_idx * c1.divisor + c1_idx)
                           / N;

    pack.storage = *(reinterpret_cast<const cuda::layer_norm::PackType<SRC, N>*>(src) + offset);
#pragma unroll
    for (int i = 0; i < N; ++i) { dst[i] = static_cast<DST>(pack.elem[i]); }
  }
  bool CanPackAs(size_t pack_size) { return (c1.divisor % pack_size) == 0; }
  const SRC* src;
  int32_t spatial_size;
  cutlass::FastDivmod c0;
  cutlass::FastDivmod c1;
};

#else

template<typename SRC, typename DST, ep::primitive::UnaryOp activation, bool affine>
struct ChannelsLastStore {
  ChannelsLastStore(DST* y, const DST* gamma, const DST* beta, int64_t spatial_size,
                    int64_t channel_size, int64_t num_groups)
      : y(y),
        gamma(gamma),
        beta(beta),
        spatial_size(spatial_size),
        c0(num_groups),
        c1(channel_size / num_groups),
        act(0, 0) {}

  template<int PackSize>
  __device__ void store(const SRC* src, int32_t row, int32_t col) {
    cuda::layer_norm::Pack<DST, PackSize> y_pack;
    cuda::layer_norm::Pack<DST, PackSize> gamma_pack;
    cuda::layer_norm::Pack<DST, PackSize> beta_pack;
    int32_t spatial_idx = col / c1;
    int32_t c1_idx = col - spatial_idx * c1;
    int32_t batch_idx = row / c0;
    int32_t c0_idx = row - batch_idx * c0;
    const int32_t y_offset =
        (batch_idx * c0 * c1 * spatial_size + spatial_idx * c0 * c1 + c0_idx * c1 + c1_idx)
        / PackSize;
    const int32_t gamma_beta_offset = (c0_idx * c1 + c1_idx) / PackSize;
    if (affine) {
      gamma_pack.storage =
          *(reinterpret_cast<const cuda::layer_norm::PackType<DST, PackSize>*>(gamma)
            + gamma_beta_offset);
      beta_pack.storage = *(reinterpret_cast<const cuda::layer_norm::PackType<DST, PackSize>*>(beta)
                            + gamma_beta_offset);
    }

#pragma unroll
    for (int i = 0; i < PackSize; ++i) {
      DST normalized_i = static_cast<DST>(src[i]);
      if (affine) {
        y_pack.elem[i] = act(normalized_i * gamma_pack.elem[i] + beta_pack.elem[i]);
      } else {
        // Direct Store.
        y_pack.elem[i] = act(normalized_i);
      }
    }
    *(reinterpret_cast<cuda::layer_norm::PackType<DST, PackSize>*>(y) + y_offset) = y_pack.storage;
  }
  bool CanPackAs(size_t pack_size) { return (c1 % pack_size) == 0; }
  DST* y;
  const DST* gamma;
  const DST* beta;
  int32_t spatial_size;
  int32_t c0;
  int32_t c1;
  ep::primitive::UnaryFunctor<DeviceType::kCUDA, activation, DST, DST> act;
};

template<typename SRC, typename DST>
struct ChannelsLastLoad {
  using LoadType = DST;
  ChannelsLastLoad(const SRC* src, int64_t spatial_size, int64_t channel_size, int64_t num_groups)
      : src(src), spatial_size(spatial_size), c0(num_groups), c1(channel_size / num_groups) {}
  template<int N>
  __device__ void load(DST* dst, int32_t row, int32_t col) const {
    int32_t spatial_idx = col / c1;
    int32_t c1_idx = col - spatial_idx * c1;
    int32_t batch_idx = row / c0;
    int32_t c0_idx = row - batch_idx * c0;
    cuda::layer_norm::Pack<SRC, N> pack;
    const int32_t offset =
        (batch_idx * c0 * c1 * spatial_size + spatial_idx * c0 * c1 + c0_idx * c1 + c1_idx) / N;

    pack.storage = *(reinterpret_cast<const cuda::layer_norm::PackType<SRC, N>*>(src) + offset);
#pragma unroll
    for (int i = 0; i < N; ++i) { dst[i] = static_cast<DST>(pack.elem[i]); }
  }
  bool CanPackAs(size_t pack_size) { return (c1 % pack_size) == 0; }
  const SRC* src;
  int32_t spatial_size;
  int32_t c0;
  int32_t c1;
};

#endif  // WITH_CUTLASS

template<typename T, ep::primitive::UnaryOp activation, bool affine>
void GroupNormForwardGpu(ep::Stream* stream, const int64_t num_instances, const int64_t norm_size,
                         const int64_t channel_size, const int64_t spatial_size,
                         const double epsilon, const T* x_ptr, const T* gamma_ptr,
                         const T* beta_ptr, T* y_ptr, user_op::Tensor* mean,
                         user_op::Tensor* inv_variance, bool channels_first) {
  using ComputeType = typename cuda::layer_norm::DefaultComputeType<T>::type;
  if (channels_first) {
    cuda::layer_norm::DirectLoad<T, T> load(x_ptr, norm_size);
    AffineStore<ComputeType, T, activation, affine> store(y_ptr, norm_size, channel_size,
                                                          spatial_size, gamma_ptr, beta_ptr);

    cuda::layer_norm::DispatchLayerNorm<decltype(load), decltype(store), ComputeType>(
        stream->As<ep::CudaStream>()->cuda_stream(), load, store, num_instances, norm_size, epsilon,
        mean->mut_dptr<ComputeType>(), inv_variance->mut_dptr<ComputeType>());
  } else {
    ChannelsLastLoad<T, T> load(x_ptr, spatial_size, channel_size,
                                channel_size / (norm_size / spatial_size));
    ChannelsLastStore<ComputeType, T, activation, affine> store(
        y_ptr, gamma_ptr, beta_ptr, spatial_size, channel_size,
        channel_size / (norm_size / spatial_size));

    cuda::layer_norm::DispatchLayerNorm<decltype(load), decltype(store), ComputeType>(
        stream->As<ep::CudaStream>()->cuda_stream(), load, store, num_instances, norm_size, epsilon,
        mean->mut_dptr<ComputeType>(), inv_variance->mut_dptr<ComputeType>());
  }
}

template<typename T, ep::primitive::UnaryOp activation>
void DispatchGroupNormAffine(ep::Stream* stream, const int64_t num_instances,
                             const int64_t norm_size, const int64_t channel_size,
                             const int64_t spatial_size, const double epsilon, const T* x_ptr,
                             const T* gamma_ptr, const T* beta_ptr, T* y_ptr, user_op::Tensor* mean,
                             user_op::Tensor* inv_variance, bool channels_first) {
  if (gamma_ptr != nullptr && beta_ptr != nullptr) {
    GroupNormForwardGpu<T, activation, true>(stream, num_instances, norm_size, channel_size,
                                             spatial_size, epsilon, x_ptr, gamma_ptr, beta_ptr,
                                             y_ptr, mean, inv_variance, channels_first);
  } else {
    GroupNormForwardGpu<T, activation, false>(stream, num_instances, norm_size, channel_size,
                                              spatial_size, epsilon, x_ptr, gamma_ptr, beta_ptr,
                                              y_ptr, mean, inv_variance, channels_first);
  }
}

template<typename T>
void DispatchGroupNormForwardGpu(ep::Stream* stream, const int64_t num_instances,
                                 const int64_t norm_size, const int64_t channel_size,
                                 const int64_t spatial_size, const double epsilon, const T* x_ptr,
                                 const T* gamma_ptr, const T* beta_ptr, T* y_ptr,
                                 user_op::Tensor* mean, user_op::Tensor* inv_variance,
                                 bool channels_first, const std::string& activation) {
  if (activation == "none") {
    DispatchGroupNormAffine<T, ep::primitive::UnaryOp::kIdentity>(
        stream, num_instances, norm_size, channel_size, spatial_size, epsilon, x_ptr, gamma_ptr,
        beta_ptr, y_ptr, mean, inv_variance, channels_first);
  } else if (activation == "silu") {
    DispatchGroupNormAffine<T, ep::primitive::UnaryOp::kSilu>(
        stream, num_instances, norm_size, channel_size, spatial_size, epsilon, x_ptr, gamma_ptr,
        beta_ptr, y_ptr, mean, inv_variance, channels_first);
  } else {
    UNIMPLEMENTED();
  }
}

template<typename T, bool affine>
void GroupNormBackwardGpu(ep::Stream* stream, const int64_t num_instances, const int64_t norm_size,
                          const int64_t channel_size, const int64_t spatial_size, const T* dy_ptr,
                          const T* x_ptr, const user_op::Tensor* mean,
                          const user_op::Tensor* inv_variance, const T* gamma_ptr, T* dx_ptr) {
  using ComputeType = typename cuda::layer_norm::DefaultComputeType<T>::type;
  cuda::layer_norm::DirectLoad<T, T> load_x(x_ptr, norm_size);
  ScaleLoad<T, T, affine> load_scaled_dy(dy_ptr, gamma_ptr, norm_size, channel_size, spatial_size);
  cuda::layer_norm::DirectStore<ComputeType, T> store(dx_ptr, norm_size);
  OF_CUDA_CHECK((cuda::layer_norm::DispatchLayerNormGrad<decltype(load_x), decltype(load_scaled_dy),
                                                         decltype(store), ComputeType>(
      stream->As<ep::CudaStream>()->cuda_stream(), load_x, load_scaled_dy, store,
      mean->dptr<ComputeType>(), inv_variance->dptr<ComputeType>(), num_instances, norm_size)));
}

template<typename T>
void LaunchGroupNormBackward(ep::Stream* stream, const int64_t num_instances,
                             const int64_t norm_size, const int64_t channel_size,
                             const int64_t spatial_size, const T* dy_ptr, const T* x_ptr,
                             const user_op::Tensor* mean, const user_op::Tensor* inv_variance,
                             const T* gamma_ptr, T* dx_ptr) {
  if (gamma_ptr != nullptr) {
    GroupNormBackwardGpu<T, true>(stream, num_instances, norm_size, channel_size, spatial_size,
                                  dy_ptr, x_ptr, mean, inv_variance, gamma_ptr, dx_ptr);
  } else {
    GroupNormBackwardGpu<T, false>(stream, num_instances, norm_size, channel_size, spatial_size,
                                   dy_ptr, x_ptr, mean, inv_variance, gamma_ptr, dx_ptr);
  }
}

}  // namespace

template<typename T>
class GroupNormGpuKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  GroupNormGpuKernel() = default;
  ~GroupNormGpuKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* mean = ctx->Tensor4ArgNameAndIndex("mean", 0);
    user_op::Tensor* inv_variance = ctx->Tensor4ArgNameAndIndex("inv_variance", 0);
    const double epsilon = ctx->Attr<double>("epsilon");
    const int32_t num_groups = ctx->Attr<int32_t>("num_groups");
    const std::string& data_format = ctx->Attr<std::string>("data_format");
    CHECK_GE(epsilon, CUDNN_BN_MIN_EPSILON);
    const int64_t num_instances = mean->shape_view().elem_cnt();  // N*num_groups
    const int64_t norm_size = x->shape_view().elem_cnt() / num_instances;
    const int64_t batch_size = x->shape_view().At(0);
    int64_t channel_size = 0;
    bool channels_first = false;
    if (data_format == "channels_first") {
      channel_size = x->shape_view().At(1);
      channels_first = true;
    } else if (data_format == "channels_last") {
      channel_size = x->shape_view().At(x->shape_view().NumAxes() - 1);
      channels_first = false;
    } else {
      UNIMPLEMENTED();
    }
    const int64_t spatial_size = x->shape_view().elem_cnt() / batch_size / channel_size;
    const T* gamma_ptr = nullptr;
    const T* beta_ptr = nullptr;
    if (ctx->has_input("gamma", 0) && ctx->has_input("beta", 0)) {
      const user_op::Tensor* gamma = ctx->Tensor4ArgNameAndIndex("gamma", 0);
      gamma_ptr = gamma->dptr<T>();
      CHECK_EQ(gamma->shape_view().elem_cnt(), channel_size);
      const user_op::Tensor* beta = ctx->Tensor4ArgNameAndIndex("beta", 0);
      beta_ptr = ctx->Tensor4ArgNameAndIndex("beta", 0)->dptr<T>();
      CHECK_EQ(beta->shape_view().elem_cnt(), channel_size);
    }
    DispatchGroupNormForwardGpu<T>(ctx->stream(), num_instances, norm_size, channel_size,
                                   spatial_size, epsilon, x->dptr<T>(), gamma_ptr, beta_ptr,
                                   y->mut_dptr<T>(), mean, inv_variance, channels_first,
                                   ctx->Attr<std::string>("activation"));
  }
};

#define REGISTER_GROUP_NORM_CUDA_KERNEL(dtype)                         \
  REGISTER_USER_KERNEL("group_norm")                                   \
      .SetCreateFn<GroupNormGpuKernel<dtype>>()                        \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_GROUP_NORM_CUDA_KERNEL(half)
REGISTER_GROUP_NORM_CUDA_KERNEL(float)
REGISTER_GROUP_NORM_CUDA_KERNEL(double)
#if CUDA_VRSION >= 11000
REGISTER_GROUP_NORM_CUDA_KERNEL(nv_bfloat16)
#endif

template<typename T>
class GroupNormGradGpuKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  GroupNormGradGpuKernel() = default;
  ~GroupNormGradGpuKernel() = default;

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
    const int64_t batch_size = x->shape_view().At(0);
    const int64_t channel_size = x->shape_view().At(1);
    const int64_t spatial_size = x->shape_view().elem_cnt() / batch_size / channel_size;
    const T* gamma_ptr = nullptr;
    if (ctx->has_input("gamma", 0)) {
      gamma_ptr = ctx->Tensor4ArgNameAndIndex("gamma", 0)->dptr<T>();
    }
    LaunchGroupNormBackward<T>(ctx->stream(), num_instances, norm_size, channel_size, spatial_size,
                               dy->dptr<T>(), x->dptr<T>(), mean, inv_variance, gamma_ptr,
                               dx->mut_dptr<T>());
  };
};

#define REGISTER_GROUP_NORM_GRAD_CUDA_KERNEL(dtype)                    \
  REGISTER_USER_KERNEL("group_norm_grad")                              \
      .SetCreateFn<GroupNormGradGpuKernel<dtype>>()                    \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value));

REGISTER_GROUP_NORM_GRAD_CUDA_KERNEL(half)
REGISTER_GROUP_NORM_GRAD_CUDA_KERNEL(float)
REGISTER_GROUP_NORM_GRAD_CUDA_KERNEL(double)
#if CUDA_VRSION >= 11000
REGISTER_GROUP_NORM_GRAD_CUDA_KERNEL(nv_bfloat16)
#endif

constexpr int kReduceBlockSize = 512;
constexpr int kBlockSize = 128;
constexpr int kNumWaves = 32;

inline cudaError_t GetReduceNumBlocks(int64_t n, int* num_blocks) {
  int dev;
  {
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) { return err; }
  }
  int sm_count;
  {
    cudaError_t err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) { return err; }
  }
  int tpm;
  {
    cudaError_t err = cudaDeviceGetAttribute(&tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
    if (err != cudaSuccess) { return err; }
  }
  *num_blocks =
      std::max<int>(1, std::min<int64_t>(n, sm_count * tpm / kReduceBlockSize * kNumWaves));
  return cudaSuccess;
}

inline cudaError_t GetNumBlocks(int64_t n, int* num_blocks) {
  int dev;
  {
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) { return err; }
  }
  int sm_count;
  {
    cudaError_t err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) { return err; }
  }
  int tpm;
  {
    cudaError_t err = cudaDeviceGetAttribute(&tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
    if (err != cudaSuccess) { return err; }
  }
  *num_blocks = std::max<int>(1, std::min<int64_t>((n + kBlockSize - 1) / kBlockSize,
                                                   sm_count * tpm / kBlockSize * kNumWaves));
  return cudaSuccess;
}

template<typename T>
struct SumOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a + b; }
};

template<typename T, int PackSize>
struct GetPackType {
  using type = typename std::aligned_storage<sizeof(T) * PackSize, sizeof(T) * PackSize>::type;
};

template<typename T, int PackSize>
using PackType = typename GetPackType<T, PackSize>::type;

template<typename T, int PackSize>
union Pack {
  static_assert(sizeof(PackType<T, PackSize>) == sizeof(T) * PackSize, "");
  __device__ Pack(T val) {
    for (int i = 0; i < PackSize; i++) { elem[i] = val; }
  }

  T elem[PackSize];
  PackType<T, PackSize> storage;
};

constexpr int kMaxPackBytes = 128 / 8;
constexpr int kMaxPackSize = 8;

constexpr int Min(int a, int b) { return a < b ? a : b; }

template<typename T>
constexpr int GetPackSize() {
  return Min(kMaxPackBytes / sizeof(T), kMaxPackSize);
}

template<typename T, typename ComputeType, int PackSize>
__global__ void GroupNormParamGradKernel(const T* dy, const T* x, const ComputeType* mean,
                                         const ComputeType* inv_var,
                                         ComputeType* dgamma_partial_sum,
                                         ComputeType* dbeta_partial_sum, const int32_t batch_size,
                                         const int32_t group_size, const int32_t channel_size,
                                         const int32_t spatial_size) {
  using LoadType = PackType<T, PackSize>;
  const int32_t batch_channel_size = batch_size * channel_size;
  for (int32_t batch_channel_id = blockIdx.x; batch_channel_id < batch_channel_size;
       batch_channel_id += gridDim.x) {
    const int32_t batch_id = batch_channel_id / channel_size;
    const int32_t channel_id = batch_channel_id % channel_size;
    const int32_t group_num = channel_size / group_size;
    const int32_t batch_group_id = batch_id * group_size + channel_id / group_num;

    ComputeType mean_val = mean[batch_group_id];
    ComputeType inv_var_val = inv_var[batch_group_id];

    Pack<ComputeType, PackSize> ds_sum_pack(0);
    Pack<ComputeType, PackSize> db_sum_pack(0);

    for (int32_t spatial = threadIdx.x * PackSize; spatial < spatial_size;
         spatial += blockDim.x * PackSize) {
      Pack<T, PackSize> dy_pack(0);
      Pack<T, PackSize> x_pack(0);
      const int32_t load_idx = batch_channel_id * spatial_size + spatial;
      const LoadType* dy_load = reinterpret_cast<const LoadType*>(dy + load_idx);
      dy_pack.storage = *dy_load;
      const LoadType* x_load = reinterpret_cast<const LoadType*>(x + load_idx);
      x_pack.storage = *x_load;
#pragma unroll
      for (int i = 0; i < PackSize; i++) {
        ds_sum_pack.elem[i] += static_cast<ComputeType>(dy_pack.elem[i])
                               * (static_cast<ComputeType>(x_pack.elem[i]) - mean_val)
                               * inv_var_val;
        db_sum_pack.elem[i] += static_cast<ComputeType>(dy_pack.elem[i]);
      }
    }

    ComputeType ds_sum = 0.0;
    ComputeType db_sum = 0.0;

#pragma unroll
    for (int i = 0; i < PackSize; i++) {
      ds_sum += ds_sum_pack.elem[i];
      db_sum += db_sum_pack.elem[i];
    }

    __syncthreads();
    typedef cub::BlockReduce<ComputeType, kReduceBlockSize> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage1;
    __shared__ typename BlockReduce::TempStorage temp_storage2;
    ComputeType ds_sum_result = BlockReduce(temp_storage1).Reduce(ds_sum, SumOp<ComputeType>());
    ComputeType db_sum_result = BlockReduce(temp_storage2).Reduce(db_sum, SumOp<ComputeType>());
    if (threadIdx.x == 0) {
      dgamma_partial_sum[batch_channel_id] = ds_sum_result;
      dbeta_partial_sum[batch_channel_id] = db_sum_result;
    }
  }
}

template<typename T, typename ComputeType>
__global__ void BatchReduceGammaBetaGradKernel(ComputeType* ds_sum, ComputeType* db_sum, T* dgamma,
                                               T* dbeta, const int32_t batch_size,
                                               const int32_t group_size, const int32_t channel_size,
                                               const int32_t spatial_size) {
  const int32_t group_num = channel_size / group_size;
  CUDA_1D_KERNEL_LOOP(channel_idx, channel_size) {
    ComputeType dgamma_sum = 0.0;
    ComputeType dbeta_sum = 0.0;
    for (int batch_id = 0; batch_id < batch_size; batch_id++) {
      const int32_t batch_group_id = batch_id * group_size + channel_idx / group_num;
      const int32_t batch_channel_id = batch_id * channel_size + channel_idx;
      dgamma_sum += ds_sum[batch_channel_id];
      dbeta_sum += db_sum[batch_channel_id];
    }
    dgamma[channel_idx] = dgamma_sum;
    dbeta[channel_idx] = dbeta_sum;
  }
}

template<typename T>
int32_t GetLaunchPackSize(const int32_t spatial_size) {
  for (int pack_size = GetPackSize<T>(); pack_size > 0; pack_size /= 2) {
    if (spatial_size % pack_size == 0) { return pack_size; }
  }
  return 1;
}

template<typename T, typename ComputeType>
void DispatchGroupNormParamGradKernel(ep::Stream* stream, const T* dy, const T* x,
                                      const ComputeType* mean, const ComputeType* inv_var,
                                      ComputeType* reduce_ds_buf, ComputeType* reduce_db_buf,
                                      const int32_t batch_size, const int32_t group_size,
                                      const int32_t channel_size, const int32_t spatial_size) {
  const int launch_pack_size = GetLaunchPackSize<T>(spatial_size);
  int num_blocks;
  OF_CUDA_CHECK(GetReduceNumBlocks(batch_size * channel_size, &num_blocks));
  if (launch_pack_size == 8) {
    GroupNormParamGradKernel<T, ComputeType, 8>
        <<<num_blocks, kReduceBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
            dy, x, mean, inv_var, reduce_ds_buf, reduce_db_buf, batch_size, group_size,
            channel_size, spatial_size);
  } else if (launch_pack_size == 4) {
    GroupNormParamGradKernel<T, ComputeType, 4>
        <<<num_blocks, kReduceBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
            dy, x, mean, inv_var, reduce_ds_buf, reduce_db_buf, batch_size, group_size,
            channel_size, spatial_size);
  } else if (launch_pack_size == 2) {
    GroupNormParamGradKernel<T, ComputeType, 2>
        <<<num_blocks, kReduceBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
            dy, x, mean, inv_var, reduce_ds_buf, reduce_db_buf, batch_size, group_size,
            channel_size, spatial_size);
  } else {
    GroupNormParamGradKernel<T, ComputeType, 1>
        <<<num_blocks, kReduceBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
            dy, x, mean, inv_var, reduce_ds_buf, reduce_db_buf, batch_size, group_size,
            channel_size, spatial_size);
  }
}

template<typename T>
class GroupNormParamGradGpuKernel final : public user_op::OpKernel,
                                          public user_op::CudaGraphSupport {
 public:
  GroupNormParamGradGpuKernel() = default;
  ~GroupNormParamGradGpuKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* mean = ctx->Tensor4ArgNameAndIndex("mean", 0);
    const user_op::Tensor* inv_variance = ctx->Tensor4ArgNameAndIndex("inv_variance", 0);
    user_op::Tensor* dgamma = ctx->Tensor4ArgNameAndIndex("dgamma", 0);
    user_op::Tensor* dbeta = ctx->Tensor4ArgNameAndIndex("dbeta", 0);
    const int64_t num_instances = mean->shape_view().elem_cnt();
    const int64_t norm_size = x->shape_view().elem_cnt() / num_instances;
    const int64_t batch_size = x->shape_view().At(0);
    const int64_t channel_size = x->shape_view().At(1);
    const int64_t spatial_size = x->shape_view().elem_cnt() / batch_size / channel_size;
    const int64_t group_size = num_instances / batch_size;
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    using ComputeType = typename cuda::layer_norm::DefaultComputeType<T>::type;
    ComputeType* reduce_ds_buf_ptr = reinterpret_cast<ComputeType*>(tmp_buffer->mut_dptr<char>());
    ComputeType* reduce_db_buf_ptr = reinterpret_cast<ComputeType*>(
        tmp_buffer->mut_dptr<char>() + batch_size * channel_size * sizeof(T));
    DispatchGroupNormParamGradKernel<T, ComputeType>(
        ctx->stream(), dy->dptr<T>(), x->dptr<T>(), mean->dptr<ComputeType>(),
        inv_variance->dptr<ComputeType>(), reduce_ds_buf_ptr, reduce_db_buf_ptr, batch_size,
        group_size, channel_size, spatial_size);
    int num_blocks;
    OF_CUDA_CHECK(GetNumBlocks(channel_size, &num_blocks));
    // Note(zhengzekang): In large batchsize, it is recommend to use gemm to reduce. (1, N) matmul
    // (N, C)
    BatchReduceGammaBetaGradKernel<T, ComputeType>
        <<<num_blocks, kBlockSize, 0, ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
            reduce_ds_buf_ptr, reduce_db_buf_ptr, dgamma->mut_dptr<T>(), dbeta->mut_dptr<T>(),
            batch_size, group_size, channel_size, spatial_size);
  };
};

#define REGISTER_GROUP_NORM_PARAM_GRAD_CUDA_KERNEL(dtype, compute_dtype)                  \
  REGISTER_USER_KERNEL("group_norm_param_grad")                                           \
      .SetCreateFn<GroupNormParamGradGpuKernel<dtype>>()                                  \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                 \
        const auto& x = ctx->InputTensorDesc("x", 0);                                     \
        const int64_t batch_size = x.shape().At(0);                                       \
        const int64_t channel_size = x.shape().At(1);                                     \
        size_t tmp_buffer_size = (2 * batch_size * channel_size) * sizeof(compute_dtype); \
        return tmp_buffer_size;                                                           \
      })                                                                                  \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                    \
                       && (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value));

REGISTER_GROUP_NORM_PARAM_GRAD_CUDA_KERNEL(half, float)
REGISTER_GROUP_NORM_PARAM_GRAD_CUDA_KERNEL(float, float)
REGISTER_GROUP_NORM_PARAM_GRAD_CUDA_KERNEL(double, double)
#if CUDA_VRSION >= 11000
REGISTER_GROUP_NORM_PARAM_GRAD_CUDA_KERNEL(nv_bfloat16, float)
#endif

}  // namespace oneflow
