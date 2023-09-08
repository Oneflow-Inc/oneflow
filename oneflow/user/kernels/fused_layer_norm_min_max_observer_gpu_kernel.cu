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
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include "oneflow/core/cuda/layer_norm.cuh"
#include "oneflow/core/cuda/layer_norm_min_max_observer.cuh"
#include "oneflow/core/ndarray/binary_func.h"
#include "oneflow/core/kernel/util/numeric_limits.cuh"
#include "oneflow/user/kernels/quantization_utils.cuh"

namespace oneflow {

namespace {

template<typename SRC, typename DST, bool do_scale, bool do_center>
struct AffineStore {
  AffineStore(DST* y, int64_t row_size, const DST* gamma, const DST* beta)
      : y(y), row_size(row_size), gamma(gamma), beta(beta) {}
  template<int N>
  __device__ void store(const SRC* src, int64_t row, int64_t col, SRC* dst) {
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
      dst[i] = y_pack.elem[i];
    }
    *(reinterpret_cast<cuda::layer_norm::PackType<DST, N>*>(y) + offset) = y_pack.storage;
  }
  DST* y;
  int64_t row_size;
  const DST* gamma;
  const DST* beta;
};

template<typename T, bool do_scale, bool do_center>
void LayerNormMinMaxObserverGpu(ep::Stream* stream, const int64_t num_instances,
                                const int64_t norm_size, const double epsilon, const T* x_ptr,
                                const T* gamma_ptr, const T* beta_ptr, T* y_ptr, T* min_max_ptr) {
  using ComputeType = typename cuda::layer_norm::DefaultComputeType<T>::type;
  cuda::layer_norm::DirectLoad<T, T> load(x_ptr, norm_size);
  AffineStore<ComputeType, T, do_scale, do_center> store(y_ptr, norm_size, gamma_ptr, beta_ptr);
  cuda::layer_norm::DispatchLayerNormMinMaxObserver<decltype(load), decltype(store), T,
                                                    ComputeType>(
      stream->As<ep::CudaStream>()->cuda_stream(), load, store, num_instances, norm_size, epsilon,
      min_max_ptr);
}

template<typename T>
void DispatchFusedLayerNormMinMaxObserverGpu(ep::Stream* stream, const int64_t num_instances,
                                             const int64_t norm_size, const double epsilon,
                                             const T* x_ptr, const T* gamma_ptr, const T* beta_ptr,
                                             T* y_ptr, T* min_max_ptr) {
  if (gamma_ptr != nullptr && beta_ptr != nullptr) {
    LayerNormMinMaxObserverGpu<T, true, true>(stream, num_instances, norm_size, epsilon, x_ptr,
                                              gamma_ptr, beta_ptr, y_ptr, min_max_ptr);
  } else if (gamma_ptr != nullptr && beta_ptr == nullptr) {
    LayerNormMinMaxObserverGpu<T, true, false>(stream, num_instances, norm_size, epsilon, x_ptr,
                                               gamma_ptr, beta_ptr, y_ptr, min_max_ptr);
  } else if (gamma_ptr == nullptr && beta_ptr != nullptr) {
    LayerNormMinMaxObserverGpu<T, false, true>(stream, num_instances, norm_size, epsilon, x_ptr,
                                               gamma_ptr, beta_ptr, y_ptr, min_max_ptr);
  } else {
    LayerNormMinMaxObserverGpu<T, false, false>(stream, num_instances, norm_size, epsilon, x_ptr,
                                                gamma_ptr, beta_ptr, y_ptr, min_max_ptr);
  }
}

template<typename T>
class GpuFusedLayerNormMinMaxObserverKernel final : public user_op::OpKernel {
 public:
  GpuFusedLayerNormMinMaxObserverKernel() = default;
  ~GpuFusedLayerNormMinMaxObserverKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const double epsilon = ctx->Attr<double>("epsilon");
    CHECK_GE(epsilon, CUDNN_BN_MIN_EPSILON);

    int64_t begin_norm_axis = ctx->Attr<int64_t>("begin_norm_axis");
    if (begin_norm_axis < 0) { begin_norm_axis += x->shape_view().NumAxes(); }
    const int64_t num_instances = x->shape_view().Count(0, begin_norm_axis);
    const int64_t norm_size = x->shape_view().elem_cnt() / num_instances;
    const T* gamma_ptr = nullptr;
    const T* beta_ptr = nullptr;
    if (ctx->has_input("gamma", 0)) {
      const user_op::Tensor* gamma = ctx->Tensor4ArgNameAndIndex("gamma", 0);
      gamma_ptr = gamma->dptr<T>();
      CHECK_EQ(gamma->shape_view().elem_cnt(), norm_size);
    }
    if (ctx->has_input("beta", 0)) { beta_ptr = ctx->Tensor4ArgNameAndIndex("beta", 0)->dptr<T>(); }

    size_t element_bytes = GetSizeOfDataType(GetDataType<T>::value);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    CHECK_GE(tmp_buffer->shape_view().elem_cnt(), num_instances * 2 * element_bytes);
    T* min_max = reinterpret_cast<T*>(tmp_buffer->mut_dptr());

    DispatchFusedLayerNormMinMaxObserverGpu<T>(ctx->stream(), num_instances, norm_size, epsilon,
                                               x->dptr<T>(), gamma_ptr, beta_ptr, y->mut_dptr<T>(),
                                               min_max);

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
            <<<1, cuda::elementwise::kBlockSize, cuda::elementwise::kBlockSize * element_bytes * 2,
               stream>>>(num_instances, min_max, upper_bound, lower_bound,
                         y_scale->mut_dptr<float>(), y_zero_point->mut_dptr<int8_t>());
      } else {
        UNIMPLEMENTED();
      }
    } else {
      UNIMPLEMENTED() << "only support oneflow quantization formula";
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_LAYER_NORM_MIN_MAX_OBSERVER_KERNEL(dtype)                       \
  REGISTER_USER_KERNEL("fused_layer_norm_min_max_observer")                            \
      .SetCreateFn<GpuFusedLayerNormMinMaxObserverKernel<dtype>>()                     \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                 \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t { return 128 * 1024 * 1024; })

REGISTER_FUSED_LAYER_NORM_MIN_MAX_OBSERVER_KERNEL(double);
REGISTER_FUSED_LAYER_NORM_MIN_MAX_OBSERVER_KERNEL(float);
REGISTER_FUSED_LAYER_NORM_MIN_MAX_OBSERVER_KERNEL(half);

}  // namespace

}  // namespace oneflow
