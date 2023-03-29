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
#include "oneflow/core/cuda/rms_norm.cuh"
#if CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#endif  // CUDA_VERSION >= 11000
#include "oneflow/core/device/cuda_pseudo_bfloat16.h"

namespace oneflow {

namespace cuda {

namespace rms_norm {

template<typename SRC, typename DST>
struct SkipLoad {
  using LoadType = DST;
  SkipLoad(const SRC* src, const SRC* bias, const SRC* skip, const float alpha, int64_t row_size)
      : src(src), bias(bias), skip(skip), alpha(alpha), row_size(row_size) {}
  template<int N>
  __device__ void load(DST* dst, int64_t row, int64_t col) const {
    layer_norm::Pack<SRC, N> src_pack;
    layer_norm::Pack<SRC, N> bias_pack;
    layer_norm::Pack<SRC, N> skip_pack;
    const int64_t offset = (row * row_size + col) / N;
    const int64_t bias_offset = col / N;
    src_pack.storage = *(reinterpret_cast<const layer_norm::PackType<SRC, N>*>(src) + offset);
    if (bias) {
      bias_pack.storage =
          *(reinterpret_cast<const layer_norm::PackType<SRC, N>*>(bias) + bias_offset);
    } else {
#pragma unroll
      for (int i = 0; i < N; ++i) { bias_pack.elem[i] = static_cast<SRC>(0.f); }
    }
    if (skip) {
      skip_pack.storage = *(reinterpret_cast<const layer_norm::PackType<SRC, N>*>(skip) + offset);
    } else {
#pragma unroll
      for (int i = 0; i < N; ++i) { skip_pack.elem[i] = static_cast<SRC>(0.f); }
    }
#pragma unroll
    for (int i = 0; i < N; ++i) {
      dst[i] = static_cast<DST>(src_pack.elem[i] + bias_pack.elem[i]
                                + skip_pack.elem[i] * static_cast<SRC>(alpha));
    }
  }
  const SRC* src;
  const SRC* bias;
  const SRC* skip;
  float alpha;
  int64_t row_size;
};

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

template<typename T, typename ComputeType, bool affine>
void DispatchSkipRmsNormForwardAffine(ep::Stream* stream, const int64_t nrow, const int64_t ncol,
                                      const double eps, const double alpha, const T* x_dptr,
                                      const T* w_dptr, const T* skip_dptr, const T* bias_dptr,
                                      T* y_dptr, ComputeType* inv_rms) {
  constexpr int32_t block_size = 128;
  unsigned int nb_element = nrow * ncol;
  unsigned int grid_size = (nb_element + block_size - 1) / block_size;
  SkipLoad<T, ComputeType> load(x_dptr, bias_dptr, skip_dptr, alpha, ncol);
  AffineStore<ComputeType, T, affine> store(y_dptr, w_dptr, ncol);
  OF_CUDA_CHECK((LaunchRmsNorm<decltype(load), decltype(store), ComputeType>(
      stream->As<ep::CudaStream>()->cuda_stream(), load, store, nrow, ncol, eps, inv_rms)));
}

template<typename T, typename ComputeType>
void SkipRmsNormForward(ep::Stream* stream, const int64_t nrow, const int64_t ncol,
                        const double eps, const double alpha, const T* x_dptr, const T* w_dptr,
                        const T* skip_dptr, const T* bias_dptr, T* y_dptr, ComputeType* inv_rms) {
  if (w_dptr) {
    DispatchSkipRmsNormForwardAffine<T, ComputeType, true>(
        stream, nrow, ncol, eps, alpha, x_dptr, w_dptr, skip_dptr, bias_dptr, y_dptr, inv_rms);
  } else {
    DispatchSkipRmsNormForwardAffine<T, ComputeType, false>(
        stream, nrow, ncol, eps, alpha, x_dptr, w_dptr, skip_dptr, bias_dptr, y_dptr, inv_rms);
  }
}

}  // namespace rms_norm

template<typename T>
class SkipRmsNormGpuKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  SkipRmsNormGpuKernel() = default;
  ~SkipRmsNormGpuKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx) const override {
    // obtain x and check its shape
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const ShapeView& x_shape = x->shape_view();
    CHECK_GE(x_shape.NumAxes(), 2)
        << "number of axes of \'x\' should be greater than or equal to 2, yet get "
        << x_shape.NumAxes();

    // obtain weight and check its shape
    const T* weight_ptr = nullptr;
    ShapeView weight_shape;
    if (ctx->has_input("weight", 0)) {
      const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
      weight_shape = weight->shape_view();
      weight_ptr = weight->dptr<T>();
      CHECK_EQ(weight_shape.NumAxes(), 1)
          << "number of axes of \'weight\' should be equal to 1, yet get "
          << weight_shape.NumAxes();
      CHECK_EQ(weight_shape.At(0), x_shape.At(x_shape.NumAxes() - 1))
          << "the size of \'weight\'(" << weight_shape.At(0)
          << ") is not consistant with the last dimension of \'x\'("
          << x_shape.At(x_shape.NumAxes() - 1) << ")";
    }

    // obtain bias and check its shape
    const T* bias_ptr = nullptr;
    ShapeView bias_shape;
    if (ctx->has_input("bias", 0)) {
      const user_op::Tensor* bias = ctx->Tensor4ArgNameAndIndex("bias", 0);
      bias_shape = bias->shape_view();
      bias_ptr = bias->dptr<T>();
      CHECK_EQ(bias_shape.NumAxes(), 1)
          << "number of axes of \'bias\' should be equal to 1, yet get " << bias_shape.NumAxes();
      CHECK_EQ(bias_shape.At(0), x_shape.At(x_shape.NumAxes() - 1))
          << "the size of \'bias\'(" << bias_shape.At(0)
          << ") is not consistant with the last dimension of \'x\'("
          << x_shape.At(x_shape.NumAxes() - 1) << ")";
    }

    // obtain skip and check its shape
    const T* skip_ptr = nullptr;
    ShapeView skip_shape;
    if (ctx->has_input("skip", 0)) {
      const user_op::Tensor* skip = ctx->Tensor4ArgNameAndIndex("skip", 0);
      skip_shape = skip->shape_view();
      skip_ptr = skip->dptr<T>();
      CHECK_EQ(skip_shape, x_shape);
    }

    // obtain epsilon and check its value
    const double epsilon = ctx->Attr<double>("epsilon");
    const double alpha = ctx->Attr<double>("alpha");

    // obtain output tensors
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* inv_rms = ctx->Tensor4ArgNameAndIndex("inv_rms", 0);
    const ShapeView& y_shape = y->shape_view();
    const ShapeView& inv_rms_shape = inv_rms->shape_view();

    // calculate number of instances and norm size
    const int64_t nrow = inv_rms->shape_view().elem_cnt();
    const int64_t ncol = x->shape_view().elem_cnt() / nrow;

    // dispatch kernel
    using ComputeType = typename layer_norm::DefaultComputeType<T>::type;
    rms_norm::SkipRmsNormForward(ctx->stream(), nrow, ncol, epsilon, alpha, x->dptr<T>(),
                                 weight_ptr, skip_ptr, bias_ptr, y->mut_dptr<T>(),
                                 inv_rms->mut_dptr<ComputeType>());
  }
};

#define REGISTER_SKIP_RMS_NORM_CUDA_KERNEL(dtype)                      \
  REGISTER_USER_KERNEL("skip_rms_norm")                                \
      .SetCreateFn<SkipRmsNormGpuKernel<dtype>>()                      \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_SKIP_RMS_NORM_CUDA_KERNEL(float)
REGISTER_SKIP_RMS_NORM_CUDA_KERNEL(double)
REGISTER_SKIP_RMS_NORM_CUDA_KERNEL(half)
#if CUDA_VERSION >= 11000
REGISTER_SKIP_RMS_NORM_CUDA_KERNEL(nv_bfloat16)
#endif

}  // namespace cuda

}  // namespace oneflow
