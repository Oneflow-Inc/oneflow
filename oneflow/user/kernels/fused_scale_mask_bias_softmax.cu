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
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/framework/user_op_tensor.h"

namespace oneflow {
namespace {
template<typename SRC, typename DST>
struct LoadWithBias {
  LoadWithBias(const SRC* x_ptr, const SRC* mask_ptr, const SRC* bias_ptr, const SRC scale,
               int64_t row_stride, int64_t bias_stride, int64_t row_size)
      : x_ptr_(x_ptr),
        mask_ptr_(mask_ptr),
        bias_ptr_(bias_ptr),
        scale_(scale),
        row_stride_(row_stride),
        bias_stride_(bias_stride),
        row_size_(row_size) {}
  template<int N>
  __device__ void load(DST* dst, int64_t row, int64_t col) const {
    cuda::softmax::Pack<SRC, N> x;
    const int64_t offset = (row * row_size_ + col) / N;
    x.storage = *(reinterpret_cast<const cuda::softmax::PackType<SRC, N>*>(x_ptr_) + offset);
    cuda::softmax::Pack<SRC, N> mask;
    const int64_t m_offset = (row / row_stride_ * row_size_ + col) / N;
    mask.storage =
        *(reinterpret_cast<const cuda::softmax::PackType<SRC, N>*>(mask_ptr_) + m_offset);
    cuda::softmax::Pack<SRC, N> bias;
    /*
    1). bias_stride_ = 0 for bias: [1, num_heads, seqlen_q, seqlen_kv]
                             x:    [batch_size, num_heads, seqlen_q, seqlen_kv]
    2). bias_stride_ > 0 for bias: [ensemble_batch, 1, num_heads, seqlen_q, seqlen_kv]
                             x:    [ensemble_batch, batch_size, num_heads, seqlen_q, seqlen_kv]
        here, bias_stride_ = batch_size, row_stride_ = num_heads * seqlen_q
        x could be viewed as [B1, B2, B3] and bias could be viewed as [B1, 1, B3] where
        B1 = ensemble_batch, B2 = batch_size = bias_stride_, B3 = num_heads * seqlen_q = row_stride_
        For row in range [0, B1 * B2 * B3) {[0, ensemble_batch * batch_size * num_heads * seqlen_q]}
        b1 = row/(B2*B3), b2=(row%(B2*B3)/B3), b3 = row%B3, after broadcast b2 will be 0 for bias.
        And finally the correspoding (broadcast) row of bias will be:
        `b1 * B3 + b3 = row/(B2*B3) * B3 + row%B3
        = row / (bias_stride_ * row_stride_) * row_stride_ + row % row_stride_`
    */
    int64_t bias_offset =
        (bias_stride_ > 0)
            ? ((row / (bias_stride_ * row_stride_) * row_stride_ + row % row_stride_) * row_size_
               + col)
                  / N
            : (row % row_stride_ * row_size_ + col) / N;
    bias.storage =
        *(reinterpret_cast<const cuda::softmax::PackType<SRC, N>*>(bias_ptr_) + bias_offset);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      dst[i] = static_cast<DST>(x.elem[i]) * static_cast<DST>(scale_)
               + static_cast<DST>(mask.elem[i]) + static_cast<DST>(bias.elem[i]);
    }
  }
  const SRC* x_ptr_;
  const SRC* mask_ptr_;
  const SRC* bias_ptr_;
  const SRC scale_;
  int64_t row_stride_;
  int64_t bias_stride_;
  int64_t row_size_;
};

template<typename SRC, typename DST>
struct LoadWithoutBias {
  LoadWithoutBias(const SRC* x_ptr, const SRC* mask_ptr, const SRC scale, int64_t row_stride,
                  int64_t row_size)
      : x_ptr_(x_ptr),
        mask_ptr_(mask_ptr),
        scale_(scale),
        row_stride_(row_stride),
        row_size_(row_size) {}
  template<int N>
  __device__ void load(DST* dst, int64_t row, int64_t col) const {
    cuda::softmax::Pack<SRC, N> x;
    const int64_t offset = (row * row_size_ + col) / N;
    x.storage = *(reinterpret_cast<const cuda::softmax::PackType<SRC, N>*>(x_ptr_) + offset);
    cuda::softmax::Pack<SRC, N> mask;
    const int64_t m_offset = (row / row_stride_ * row_size_ + col) / N;
    mask.storage =
        *(reinterpret_cast<const cuda::softmax::PackType<SRC, N>*>(mask_ptr_) + m_offset);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      dst[i] =
          static_cast<DST>(x.elem[i]) * static_cast<DST>(scale_) + static_cast<DST>(mask.elem[i]);
    }
  }
  const SRC* x_ptr_;
  const SRC* mask_ptr_;
  const SRC scale_;
  int64_t row_stride_;
  int64_t row_size_;
};

template<typename T, typename ComputeType = typename cuda::softmax::DefaultComputeType<T>::type>
void LaunchFusedSoftmaxForwardKernel(cudaStream_t stream, T* out, const T* x, const T* mask,
                                     const T* bias, T scale, const int64_t row_stride,
                                     const int64_t bias_stride, const int64_t rows,
                                     const int64_t row_size) {
  cuda::softmax::DirectStore<ComputeType, T> store(out, row_size);
  if (bias != nullptr) {
    LoadWithBias<T, ComputeType> load(x, mask, bias, scale, row_stride, bias_stride, row_size);
    OF_CUDA_CHECK((cuda::softmax::DispatchSoftmax<decltype(load), decltype(store), ComputeType>(
        stream, load, store, rows, row_size)));
  } else {
    LoadWithoutBias<T, ComputeType> load(x, mask, scale, row_stride, row_size);
    OF_CUDA_CHECK((cuda::softmax::DispatchSoftmax<decltype(load), decltype(store), ComputeType>(
        stream, load, store, rows, row_size)));
  }
};

template<typename SRC, typename DST>
struct GradStore {
  GradStore(DST* dx, const SRC scale, int64_t row_size)
      : dx(dx), scale(scale), row_size(row_size) {}
  template<int N>
  __device__ void store(const SRC* dout, int64_t row, int64_t col) const {
    cuda::softmax::Pack<DST, N> x;
    const int64_t offset = (row * row_size + col) / N;
#pragma unroll
    for (int i = 0; i < N; ++i) { x.elem[i] = static_cast<DST>(dout[i]) * static_cast<DST>(scale); }
    *(reinterpret_cast<cuda::softmax::PackType<DST, N>*>(dx) + offset) = x.storage;
  }
  DST* dx;
  const SRC scale;
  int64_t row_size;
};

template<typename T, typename ComputeType = typename cuda::softmax::DefaultComputeType<T>::type>
void LaunchSoftmaxBackwardKernel(cudaStream_t stream, T* dx, const T* y, const T* dy, T scale,
                                 const int64_t rows, const int64_t row_size) {
  GradStore<ComputeType, T> store(dx, scale, row_size);
  cuda::softmax::DirectLoad<T, ComputeType> load_y(y, row_size);
  cuda::softmax::DirectLoad<T, ComputeType> load_dy(dy, row_size);
  OF_CUDA_CHECK((cuda::softmax::DispatchSoftmaxGrad<decltype(load_y), decltype(load_dy),
                                                    decltype(store), ComputeType>(
      stream, load_y, load_dy, store, rows, row_size)));
};

}  // namespace

template<typename T>
class FusedScaleMaskBiasSoftmaxKernel final : public user_op::OpKernel {
 public:
  FusedScaleMaskBiasSoftmaxKernel() = default;
  ~FusedScaleMaskBiasSoftmaxKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* mask = ctx->Tensor4ArgNameAndIndex("mask", 0);
    const T scale = ctx->Attr<float>("scale");
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    auto x_shape = x->shape_view();
    auto axes = x_shape.NumAxes();
    /*
     * axes=3 for x: [batch_size, num_heads, seq], mask: [batch_size, 1, seq], no bias here
     * axes=4 for x: [batch_size, num_heads, seq_len_q, seq_len_kv]
     *            mask: [batch_size, 1, 1, seq_len_kv]
     *            bias: [1, num_heads, seq_len_q, seq_len_kv]
     * axes=5 for x: [ensemble_batch, batch_size, num_heads, seq_len_q, seq_len_kv]
     *            mask: [ensemble_batch, batch_size, 1, 1, seq_len_kv]
     *            bias: [ensemble_batch, 1, num_heads, seq_len_q, seq_len_kv]
     * `axes=5` is equivalent to `axes=4` when ensemble_batch = 1 .
     *
     * row_stride is used for computing `mask` stride and
     * bias_stride for computing `bias` stride
     * row_stride is num_heads (for `axes=3`) or num_heads * seq_len_q (for `axes=4` & `axes=5`)
     * bias_stride is 0 (for `axes=4`) or batch_size (for `axes=5`)
     * row_size = seq_len_k (the last dimension of `x`)
     */
    CHECK(axes == 3 || axes == 4 || axes == 5);
    auto mask_shape = mask->shape_view();
    CHECK(mask_shape.NumAxes() == axes);
    const int row_size = x_shape.At(axes - 1);
    const int rows = x_shape.elem_cnt() / row_size;
    int row_stride = 1;
    for (int i = axes - 2; i >= 0; i--) {
      if (mask_shape.At(i) == 1)
        row_stride *= x_shape.At(i);
      else
        break;
    }

    user_op::Tensor* bias = nullptr;
    int64_t bias_stride = 0;
    if (ctx->has_input("bias", 0)) {
      bias = ctx->Tensor4ArgNameAndIndex("bias", 0);
      if (axes == 5 && x_shape.At(0) != 1) bias_stride = x_shape.At(1);
    }
    LaunchFusedSoftmaxForwardKernel<T>(ctx->stream()->As<ep::CudaStream>()->cuda_stream(),
                                       out->mut_dptr<T>(), x->dptr<T>(), mask->dptr<T>(),
                                       ctx->has_input("bias", 0) ? bias->dptr<T>() : nullptr, scale,
                                       row_stride, bias_stride, rows, row_size);
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_SCALE_MASK_BIAS_SOFTMAX_KERNEL_GPU(dtype)       \
  REGISTER_USER_KERNEL("fused_scale_mask_bias_softmax")                \
      .SetCreateFn<FusedScaleMaskBiasSoftmaxKernel<dtype>>()           \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_FUSED_SCALE_MASK_BIAS_SOFTMAX_KERNEL_GPU(half)
#if CUDA_VERSION >= 11000
REGISTER_FUSED_SCALE_MASK_BIAS_SOFTMAX_KERNEL_GPU(nv_bfloat16)
#endif
REGISTER_FUSED_SCALE_MASK_BIAS_SOFTMAX_KERNEL_GPU(float)

template<typename T>
class FusedScaleMaskBiasSoftmaxGradKernel final : public user_op::OpKernel {
 public:
  FusedScaleMaskBiasSoftmaxGradKernel() = default;
  ~FusedScaleMaskBiasSoftmaxGradKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const T scale = ctx->Attr<float>("scale");
    auto y_shape = y->shape_view();

    const int64_t axes = y_shape.NumAxes();
    int64_t row_size = y_shape.At(axes - 1);
    int64_t rows = y_shape.elem_cnt() / row_size;

    LaunchSoftmaxBackwardKernel<T>(ctx->stream()->As<ep::CudaStream>()->cuda_stream(),
                                   dx->mut_dptr<T>(), y->dptr<T>(), dy->dptr<T>(), scale, rows,
                                   row_size);
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_SCALE_MASK_BIAS_SOFTMAX_GRAD_KERNEL_GPU(dtype)  \
  REGISTER_USER_KERNEL("fused_scale_mask_bias_softmax_grad")           \
      .SetCreateFn<FusedScaleMaskBiasSoftmaxGradKernel<dtype>>()       \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_FUSED_SCALE_MASK_BIAS_SOFTMAX_GRAD_KERNEL_GPU(half)
#if CUDA_VERSION >= 11000
REGISTER_FUSED_SCALE_MASK_BIAS_SOFTMAX_GRAD_KERNEL_GPU(nv_bfloat16)
#endif
REGISTER_FUSED_SCALE_MASK_BIAS_SOFTMAX_GRAD_KERNEL_GPU(float)

}  // namespace oneflow
