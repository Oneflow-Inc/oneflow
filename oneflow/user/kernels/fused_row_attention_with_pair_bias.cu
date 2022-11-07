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
#include "oneflow/core/cuda/softmax.cuh"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/framework/user_op_tensor.h"

namespace oneflow {
namespace cuda {
namespace softmax {
template<typename SRC, typename DST = SRC>
struct MSADirectLoad {
  MSADirectLoad(const SRC* q, const SRC* m, const SRC* p, const SRC scale, int64_t stride,
                int64_t row_size)
      : q(q), m(m), p(p), scale(scale), stride(stride), row_size(row_size) {}
  template<int N>
  __device__ void load(DST* dst, int64_t row, int64_t col) const {
    Pack<SRC, N> qmk;
    const int64_t offset = (row * row_size + col) / N;
    qmk.storage = *(reinterpret_cast<const PackType<SRC, N>*>(q) + offset);  // BhS * S2
    Pack<SRC, N> mask;
    const int64_t m_offset = (row / stride * row_size + col) / N;               // B * S, stride=h*S
    mask.storage = *(reinterpret_cast<const PackType<SRC, N>*>(m) + m_offset);  // BhS * S2
    Pack<SRC, N> pair_bias;
    const int64_t p_offset = (row % stride * row_size + col) / N;  // hS * S, stride=h*S
    pair_bias.storage = *(reinterpret_cast<const PackType<SRC, N>*>(p) + p_offset);  // BhS * S2
#pragma unroll
    for (int i = 0; i < N; ++i) {
      dst[i] = static_cast<DST>(qmk.elem[i] * scale) + static_cast<DST>(mask.elem[i])
               + static_cast<DST>(pair_bias.elem[i]);
    }
  }
  const SRC* q;
  const SRC* m;
  const SRC* p;
  const SRC scale;
  int64_t stride;
  int64_t row_size;
};

template<typename T, typename ComputeType>
void LaunchMSABroadcastForwardKernel(cudaStream_t stream, T* out, const T* qmk, const T* mask,
                                     const T* pair_bias, T scale, const int64_t stride,
                                     const int64_t row_size, const int64_t rows,
                                     const int64_t cols) {
  DirectStore<ComputeType, T> store(out, row_size);
  MSADirectLoad<T, ComputeType> load(qmk, mask, pair_bias, scale, stride, row_size);
  OF_CUDA_CHECK((DispatchSoftmax<decltype(load), decltype(store), ComputeType>(stream, load, store,
                                                                               rows, cols)));
}

template<typename SRC, typename DST = SRC>
struct MSAGradDirectStore {
  MSAGradDirectStore(DST* dx, DST* dp, const SRC scale, int64_t stride, int64_t row_size)
      : dx(dx), dp(dp), scale(scale), stride(stride), row_size(row_size) {}
  template<int N>
  __device__ void store(const SRC* dout, int64_t row, int64_t col) const {
    Pack<SRC, N> qmk;
    Pack<SRC, N> pair;
    const int64_t offset = (row * row_size + col) / N;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      qmk.elem[i] = static_cast<DST>(dout[i] * scale);
      pair.elem[i] = static_cast<DST>(dout[i]);
    }
    *(reinterpret_cast<PackType<DST, N>*>(dx) + offset) = qmk.storage;
    *(reinterpret_cast<PackType<DST, N>*>(dp) + offset) = pair.storage;
  }
  SRC* dx;
  SRC* dp;
  const SRC scale;
  int64_t stride;
  int64_t row_size;
};

template<typename T, typename ComputeType>
void LaunchMSABroadcastBackwardKernel(cudaStream_t stream, T* dx, T* dp, const T* y, const T* dy,
                                      T scale, const int64_t stride, const int64_t row_size,
                                      const int64_t rows, const int64_t cols) {
  MSAGradDirectStore<ComputeType, T> store(dx, dp, scale, stride, row_size);
  DirectLoad<T, ComputeType> load_y(y, row_size);
  DirectLoad<T, ComputeType> load_dy(dy, row_size);
  OF_CUDA_CHECK(
      (DispatchSoftmaxGrad<decltype(load_y), decltype(load_dy), decltype(store), ComputeType>(
          stream, load_y, load_dy, store, rows, cols)));
}
}  // namespace softmax

}  // namespace cuda

namespace {

/* fused MSARowAttentionWithPairBias
 * softmax(qmk/scale + mask + pair_bias)
 * qmk :       B, h, S, S
 * mask_bias : B, S
 * pair_bias : h, S, S
 * scale     : rsqrt(head_size)
 */
};  // namespace
template<typename T>
class FusedRowAttentionWithPairBiasKernel final : public user_op::OpKernel {
 public:
  FusedRowAttentionWithPairBiasKernel() = default;
  ~FusedRowAttentionWithPairBiasKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* qmk = ctx->Tensor4ArgNameAndIndex("qmk", 0);
    const user_op::Tensor* mask_bias = ctx->Tensor4ArgNameAndIndex("mask_bias", 0);
    const user_op::Tensor* pair_bias = ctx->Tensor4ArgNameAndIndex("pair_bias", 0);
    const T scale = ctx->Attr<T>("scale");
    const T eps = ctx->Attr<T>("eps");
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    auto qmk_shape = qmk->shape_view();
    const int64_t elem_cnt = qmk_shape.elem_cnt();
    auto out_shape = out->shape_view();

    const int64_t B = qmk_shape.At(0), h = qmk_shape.At(1), S = qmk_shape.At(2);
    cuda::softmax::LaunchMSABroadcastForwardKernel<T, T>(
        ctx->stream()->As<ep::CudaStream>()->cuda_stream(), out->mut_dptr<T>(), qmk->dptr<T>(),
        mask_bias->dptr<T>(), pair_bias->dptr<T>(), scale, h * S, S, B * h * S, S);
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_ROW_ATTENTION_WITH_PAIR_BIAS_KERNEL_GPU(dtype)  \
  REGISTER_USER_KERNEL("fused_row_attention_with_pair_bias")           \
      .SetCreateFn<FusedRowAttentionWithPairBiasKernel<dtype>>()       \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_FUSED_ROW_ATTENTION_WITH_PAIR_BIAS_KERNEL_GPU(float)
REGISTER_FUSED_ROW_ATTENTION_WITH_PAIR_BIAS_KERNEL_GPU(double)

template<typename T>
class FusedRowAttentionWithPairBiasGradKernel final : public user_op::OpKernel {
 public:
  FusedRowAttentionWithPairBiasGradKernel() = default;
  ~FusedRowAttentionWithPairBiasGradKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    user_op::Tensor* dp = ctx->Tensor4ArgNameAndIndex("dp", 0);
    const T scale = ctx->Attr<T>("scale");
    const int64_t stride = ctx->Attr<int64_t>("stride");
    auto y_shape = y->shape_view();

    const int64_t B = y_shape.At(0), h = y_shape.At(1), S = y_shape.At(2);
    cuda::softmax::LaunchMSABroadcastBackwardKernel<T, T>(
        ctx->stream()->As<ep::CudaStream>()->cuda_stream(), dx->mut_dptr<T>(), dp->mut_dptr<T>(),
        y->dptr<T>(), dy->dptr<T>(), scale, h * S, S, B * h * S, S);
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_ROW_ATTENTION_WITH_PAIR_BIAS_GRAD_KERNEL_GPU(dtype)             \
  REGISTER_USER_KERNEL("fused_row_attention_with_pair_bias_grad")                      \
      .SetCreateFn<FusedRowAttentionWithPairBiasGradKernel<dtype>>()                   \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                 \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobDataType("dp", 0) == GetDataType<dtype>::value));

REGISTER_FUSED_ROW_ATTENTION_WITH_PAIR_BIAS_GRAD_KERNEL_GPU(float)
REGISTER_FUSED_ROW_ATTENTION_WITH_PAIR_BIAS_GRAD_KERNEL_GPU(double)
}  // namespace oneflow
