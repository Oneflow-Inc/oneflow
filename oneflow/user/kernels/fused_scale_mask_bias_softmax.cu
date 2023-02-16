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
  LoadWithBias(const SRC* q, const SRC* m, const SRC* p, const SRC scale, int64_t stride,
               int64_t row_size)
      : q(q), m(m), p(p), scale(scale), stride(stride), row_size(row_size) {}
  template<int N>
  __device__ void load(DST* dst, int64_t row, int64_t col) const {
    cuda::softmax::Pack<SRC, N> x;
    const int64_t offset = (row * row_size + col) / N;
    x.storage = *(reinterpret_cast<const cuda::softmax::PackType<SRC, N>*>(q) + offset);
    cuda::softmax::Pack<SRC, N> mask;
    const int64_t m_offset = (row / stride * row_size + col) / N;
    mask.storage = *(reinterpret_cast<const cuda::softmax::PackType<SRC, N>*>(m) + m_offset);
    cuda::softmax::Pack<SRC, N> pair_bias;
    const int64_t p_offset = (row % stride * row_size + col) / N;
    pair_bias.storage = *(reinterpret_cast<const cuda::softmax::PackType<SRC, N>*>(p) + p_offset);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      dst[i] = static_cast<DST>(x.elem[i]) * static_cast<DST>(scale)
               + static_cast<DST>(mask.elem[i]) + static_cast<DST>(pair_bias.elem[i]);
    }
  }
  const SRC* q;
  const SRC* m;
  const SRC* p;
  const SRC scale;
  int64_t stride;
  int64_t row_size;
};

template<typename SRC, typename DST>
struct LoadWithoutBias {
  LoadWithoutBias(const SRC* q, const SRC* m, const SRC scale, int64_t stride, int64_t row_size)
      : q(q), m(m), scale(scale), stride(stride), row_size(row_size) {}
  template<int N>
  __device__ void load(DST* dst, int64_t row, int64_t col) const {
    cuda::softmax::Pack<SRC, N> x;
    const int64_t offset = (row * row_size + col) / N;
    x.storage = *(reinterpret_cast<const cuda::softmax::PackType<SRC, N>*>(q) + offset);
    cuda::softmax::Pack<SRC, N> mask;
    const int64_t m_offset = (row / stride * row_size + col) / N;
    mask.storage = *(reinterpret_cast<const cuda::softmax::PackType<SRC, N>*>(m) + m_offset);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      dst[i] =
          static_cast<DST>(x.elem[i]) * static_cast<DST>(scale) + static_cast<DST>(mask.elem[i]);
    }
  }
  const SRC* q;
  const SRC* m;
  const SRC scale;
  int64_t stride;
  int64_t row_size;
};

template<typename T, typename ComputeType = typename cuda::softmax::DefaultComputeType<T>::type>
void LaunchLoadWithBiasSoftmaxForwardKernel(cudaStream_t stream, T* out, const T* x, const T* mask,
                                            const T* bias, T scale, const int64_t stride,
                                            const int64_t row_size, const int64_t rows,
                                            const int64_t cols) {
  cuda::softmax::DirectStore<ComputeType, T> store(out, row_size);
  LoadWithBias<T, ComputeType> load(x, mask, bias, scale, stride, row_size);
  OF_CUDA_CHECK((cuda::softmax::DispatchSoftmax<decltype(load), decltype(store), ComputeType>(
      stream, load, store, rows, cols)));
};

template<typename T, typename ComputeType = typename cuda::softmax::DefaultComputeType<T>::type>
void LaunchLoadWithoutBiasSoftmaxForwardKernel(cudaStream_t stream, T* out, const T* x,
                                               const T* mask, T scale, const int64_t stride,
                                               const int64_t row_size, const int64_t rows,
                                               const int64_t cols) {
  cuda::softmax::DirectStore<ComputeType, T> store(out, row_size);
  LoadWithoutBias<T, ComputeType> load(x, mask, scale, stride, row_size);
  OF_CUDA_CHECK((cuda::softmax::DispatchSoftmax<decltype(load), decltype(store), ComputeType>(
      stream, load, store, rows, cols)));
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
                                 const int64_t row_size, const int64_t rows, const int64_t cols) {
  GradStore<ComputeType, T> store(dx, scale, row_size);
  cuda::softmax::DirectLoad<T, ComputeType> load_y(y, row_size);
  cuda::softmax::DirectLoad<T, ComputeType> load_dy(dy, row_size);
  OF_CUDA_CHECK((
      cuda::softmax::DispatchSoftmaxGrad<decltype(load_y), decltype(load_dy), decltype(store),
                                         ComputeType>(stream, load_y, load_dy, store, rows, cols)));
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
    const std::string mode = ctx->Attr<std::string>("mode");
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    auto x_shape = x->shape_view();
    auto axes = x_shape.NumAxes();
    int64_t B = x_shape.At(0), h = x_shape.At(1), S1 = x_shape.At(2), S2 = x_shape.At(axes - 1);
    if (mode == "template") {
      B = x_shape.At(0) * x_shape.At(1);
      h = x_shape.At(2);
      S1 = x_shape.At(3);
      S2 = x_shape.At(4);
    } else if (axes == 5) {
      CHECK_EQ(x_shape.At(0), 1);
      B = x_shape.At(1);
      h = x_shape.At(2);
      S1 = x_shape.At(3);
      S2 = x_shape.At(4);
    }

    if (ctx->has_input("bias", 0)) {
      const user_op::Tensor* bias = ctx->Tensor4ArgNameAndIndex("bias", 0);
      LaunchLoadWithBiasSoftmaxForwardKernel<T>(ctx->stream()->As<ep::CudaStream>()->cuda_stream(),
                                                out->mut_dptr<T>(), x->dptr<T>(), mask->dptr<T>(),
                                                bias->dptr<T>(), scale, h * S1, S2, B * h * S1, S2);
    } else {
      int64_t stride = mode == "template" ? B * h * S1 : (mode == "col" ? h * S1 : h);
      int64_t rows = mode == "global_col" ? h * B : B * h * S1;
      LaunchLoadWithoutBiasSoftmaxForwardKernel<T>(
          ctx->stream()->As<ep::CudaStream>()->cuda_stream(), out->mut_dptr<T>(), x->dptr<T>(),
          mask->dptr<T>(), scale, stride, S2, rows, S2);
    }
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
    const std::string mode = ctx->Attr<std::string>("mode");
    auto y_shape = y->shape_view();

    const int64_t axes = y_shape.NumAxes();
    int64_t rows = y_shape.At(0) * y_shape.At(1), S = y_shape.At(axes - 1);
    rows = mode == "global_col" ? rows : rows * y_shape.At(2);

    LaunchSoftmaxBackwardKernel<T>(ctx->stream()->As<ep::CudaStream>()->cuda_stream(),
                                   dx->mut_dptr<T>(), y->dptr<T>(), dy->dptr<T>(), scale, S, rows,
                                   S);
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
