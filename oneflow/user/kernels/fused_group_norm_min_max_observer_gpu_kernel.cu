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
#include "oneflow/core/common/util.h"
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/ep/include/primitive/unary_op.h"
#include "oneflow/core/ep/cuda/primitive/unary_functor.cuh"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_kernel.h"
#include "oneflow/core/framework/user_op_tensor.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include "oneflow/core/cuda/layer_norm.cuh"
#include "oneflow/core/cuda/layer_norm_min_max_observer.cuh"
#include "oneflow/core/ndarray/binary_func.h"
#include "oneflow/core/kernel/util/numeric_limits.cuh"
#include "oneflow/user/kernels/quantization_utils.cuh"

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
  __device__ void store(const SRC* src, int64_t row, int64_t col, SRC* dst) {
    cuda::layer_norm::Pack<DST, PackSize> y_pack;
    cuda::layer_norm::Pack<DST, PackSize> gamma_pack;
    cuda::layer_norm::Pack<DST, PackSize> beta_pack;
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
        y_pack.elem[i] = act(normalized_i);
      }
      dst[i] = y_pack.elem[i];
    }
    *(reinterpret_cast<cuda::layer_norm::PackType<DST, PackSize>*>(y) + packed_offset) =
        y_pack.storage;
  }
  DST* y;
  int64_t row_size;
  int64_t channel_size;
  int64_t spatial_size;
  const DST* gamma;
  const DST* beta;
  ep::primitive::UnaryFunctor<DeviceType::kCUDA, activation, DST, DST> act;
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
  __device__ void store(const SRC* src, int32_t row, int32_t col, SRC* dst) {
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
    } else {
#pragma unroll
      for (int i = 0; i < PackSize; ++i) {
        gamma_pack.elem[i] = static_cast<DST>(1.f);
        beta_pack.elem[i] = static_cast<DST>(0.f);
      }
    }

#pragma unroll
    for (int i = 0; i < PackSize; ++i) {
      DST normalized_i = static_cast<DST>(src[i]);
      if (affine) {
        y_pack.elem[i] = act(normalized_i * gamma_pack.elem[i] + beta_pack.elem[i]);
      } else {
        y_pack.elem[i] = act(normalized_i);
      }
      dst[i] = y_pack.elem[i];
    }
    *(reinterpret_cast<cuda::layer_norm::PackType<DST, PackSize>*>(y) + y_offset) = y_pack.storage;
  }
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
  __device__ void store(const SRC* src, int32_t row, int32_t col, SRC* dst) {
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
    } else {
#pragma unroll
      for (int i = 0; i < PackSize; ++i) {
        gamma_pack.elem[i] = static_cast<DST>(1.f);
        beta_pack.elem[i] = static_cast<DST>(0.f);
      }
    }

#pragma unroll
    for (int i = 0; i < PackSize; ++i) {
      DST normalized_i = static_cast<DST>(src[i]);
      if (affine) {
        y_pack.elem[i] = act(normalized_i * gamma_pack.elem[i] + beta_pack.elem[i]);
      } else {
        y_pack.elem[i] = act(normalized_i);
      }
    }
    *(reinterpret_cast<cuda::layer_norm::PackType<DST, PackSize>*>(y) + y_offset) = y_pack.storage;
  }
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
void GroupNormMinMaxObserverForward(ep::Stream* stream, int64_t num_instances, int64_t norm_size,
                                    int64_t channel_size, int64_t spatial_size,
                                    const double epsilon, const T* x_ptr, const T* gamma_ptr,
                                    const T* beta_ptr, T* y_ptr, bool channels_first,
                                    T* min_max_ptr) {
  using ComputeType = typename cuda::layer_norm::DefaultComputeType<T>::type;
  if (channels_first) {
    cuda::layer_norm::DirectLoad<T, T> load(x_ptr, norm_size);
    AffineStore<ComputeType, T, activation, affine> store(y_ptr, norm_size, channel_size,
                                                          spatial_size, gamma_ptr, beta_ptr);
    cuda::layer_norm::DispatchLayerNormMinMaxObserver<decltype(load), decltype(store), T,
                                                      ComputeType>(
        stream->As<ep::CudaStream>()->cuda_stream(), load, store, num_instances, norm_size, epsilon,
        min_max_ptr);
  } else {
    ChannelsLastLoad<T, T> load(x_ptr, spatial_size, channel_size,
                                channel_size / (norm_size / spatial_size));
    ChannelsLastStore<ComputeType, T, activation, affine> store(
        y_ptr, gamma_ptr, beta_ptr, spatial_size, channel_size,
        channel_size / (norm_size / spatial_size));
    cuda::layer_norm::DispatchLayerNormMinMaxObserver<decltype(load), decltype(store), T,
                                                      ComputeType>(
        stream->As<ep::CudaStream>()->cuda_stream(), load, store, num_instances, norm_size, epsilon,
        min_max_ptr);
  }
}

template<typename T, ep::primitive::UnaryOp activation>
void DispatchFusedGroupNormMinMaxObserverAffine(ep::Stream* stream, const int64_t num_instances,
                                                const int64_t norm_size, int64_t channel_size,
                                                int64_t spatial_size, const double epsilon,
                                                const T* x_ptr, const T* gamma_ptr,
                                                const T* beta_ptr, T* y_ptr, bool channels_first,
                                                T* min_max_ptr) {
  if (gamma_ptr != nullptr && beta_ptr != nullptr) {
    GroupNormMinMaxObserverForward<T, activation, true>(
        stream, num_instances, norm_size, channel_size, spatial_size, epsilon, x_ptr, gamma_ptr,
        beta_ptr, y_ptr, channels_first, min_max_ptr);
  } else {
    GroupNormMinMaxObserverForward<T, activation, false>(
        stream, num_instances, norm_size, channel_size, spatial_size, epsilon, x_ptr, gamma_ptr,
        beta_ptr, y_ptr, channels_first, min_max_ptr);
  }
}

template<typename T>
void DispatchFusedGroupNormMinMaxObserverActivation(ep::Stream* stream, const int64_t num_instances,
                                                    const int64_t norm_size, int64_t channel_size,
                                                    int64_t spatial_size, const double epsilon,
                                                    const T* x_ptr, const T* gamma_ptr,
                                                    const T* beta_ptr, T* y_ptr,
                                                    bool channels_first,
                                                    const std::string& activation, T* min_max_ptr) {
  if (activation == "none") {
    DispatchFusedGroupNormMinMaxObserverAffine<T, ep::primitive::UnaryOp::kIdentity>(
        stream, num_instances, norm_size, channel_size, spatial_size, epsilon, x_ptr, gamma_ptr,
        beta_ptr, y_ptr, channels_first, min_max_ptr);
  } else if (activation == "silu") {
    DispatchFusedGroupNormMinMaxObserverAffine<T, ep::primitive::UnaryOp::kSilu>(
        stream, num_instances, norm_size, channel_size, spatial_size, epsilon, x_ptr, gamma_ptr,
        beta_ptr, y_ptr, channels_first, min_max_ptr);
  } else {
    UNIMPLEMENTED();
  }
}

template<typename T>
class CUDAFusedGroupNormMinMaxObserverKernel final : public user_op::OpKernel {
 public:
  CUDAFusedGroupNormMinMaxObserverKernel() = default;
  ~CUDAFusedGroupNormMinMaxObserverKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const double epsilon = ctx->Attr<double>("epsilon");
    const int32_t num_groups = ctx->Attr<int32_t>("num_groups");
    const std::string& data_format = ctx->Attr<std::string>("data_format");
    const std::string& activation = ctx->Attr<std::string>("activation");
    CHECK_GE(epsilon, CUDNN_BN_MIN_EPSILON);

    const int64_t num_instances = x->shape_view().At(0) * num_groups;
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
      beta_ptr = beta->dptr<T>();
      CHECK_EQ(beta->shape_view().elem_cnt(), channel_size);
    }
    size_t element_bytes = GetSizeOfDataType(GetDataType<T>::value);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    CHECK_GE(tmp_buffer->shape_view().elem_cnt(), num_instances * 2 * element_bytes);
    T* min_max = reinterpret_cast<T*>(tmp_buffer->mut_dptr());

    DispatchFusedGroupNormMinMaxObserverActivation<T>(
        ctx->stream(), num_instances, norm_size, channel_size, spatial_size, epsilon, x->dptr<T>(),
        gamma_ptr, beta_ptr, y->mut_dptr<T>(), channels_first, activation, min_max);

    // quantization part
    const std::string quantization_scheme = ctx->Attr<std::string>("quantization_scheme");
    const int32_t quantization_bit = ctx->Attr<int32_t>("quantization_bit");
    const std::string quantization_formula = ctx->Attr<std::string>("quantization_formula");
    CHECK(quantization_scheme == "affine");

    user_op::Tensor* y_scale = ctx->Tensor4ArgNameAndIndex("y_scale", 0);
    user_op::Tensor* y_zero_point = ctx->Tensor4ArgNameAndIndex("y_zero_point", 0);

    auto stream = ctx->stream()->As<ep::CudaStream>()->cuda_stream();
    if (quantization_formula == "oneflow") {
      if (quantization_bit == 8) {
        int8_t upper_bound = (1 << (quantization_bit - 1)) - 1;
        int8_t lower_bound = -upper_bound - 1;
        quantization::ComputeScaleAndZeroPointBlock<T, int8_t>
            <<<1, cuda::elementwise::kBlockSize, 0, stream>>>(
                num_instances, min_max, upper_bound, lower_bound, y_scale->mut_dptr<float>(),
                y_zero_point->mut_dptr<int8_t>());
      } else {
        UNIMPLEMENTED();
      }
    } else {
      UNIMPLEMENTED() << "only support oneflow quantization formula";
    }
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_GROUP_NORM_MIN_MAX_OBSERVER_KERNEL(dtype)                       \
  REGISTER_USER_KERNEL("fused_group_norm_min_max_observer")                            \
      .SetCreateFn<CUDAFusedGroupNormMinMaxObserverKernel<dtype>>()                    \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                 \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t { return 128 * 1024 * 1024; })

REGISTER_FUSED_GROUP_NORM_MIN_MAX_OBSERVER_KERNEL(double);
REGISTER_FUSED_GROUP_NORM_MIN_MAX_OBSERVER_KERNEL(float);
REGISTER_FUSED_GROUP_NORM_MIN_MAX_OBSERVER_KERNEL(half);

}  // namespace
}  // namespace oneflow
