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
namespace alphafold {
namespace attn {
template<typename SRC, typename DST>
struct MSALoadWithBias {
  MSALoadWithBias(const SRC* q, const SRC* m, const SRC* p, const SRC scale, int64_t stride,
                  int64_t row_size)
      : q(q), m(m), p(p), scale(scale), stride(stride), row_size(row_size) {}
  template<int N>
  __device__ void load(DST* dst, int64_t row, int64_t col) const {
    cuda::softmax::Pack<SRC, N> qmk;
    const int64_t offset = (row * row_size + col) / N;
    qmk.storage = *(reinterpret_cast<const cuda::softmax::PackType<SRC, N>*>(q) + offset);
    cuda::softmax::Pack<SRC, N> mask;
    const int64_t m_offset = (row / stride * row_size + col) / N;
    mask.storage = *(reinterpret_cast<const cuda::softmax::PackType<SRC, N>*>(m) + m_offset);
    cuda::softmax::Pack<SRC, N> pair_bias;
    const int64_t p_offset = (row % stride * row_size + col) / N;
    pair_bias.storage = *(reinterpret_cast<const cuda::softmax::PackType<SRC, N>*>(p) + p_offset);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      dst[i] = static_cast<DST>(qmk.elem[i]) * static_cast<DST>(scale)
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
struct MSALoad {
  MSALoad(const SRC* q, const SRC* m, const SRC scale, int64_t stride, int64_t row_size)
      : q(q), m(m), scale(scale), stride(stride), row_size(row_size) {}
  template<int N>
  __device__ void load(DST* dst, int64_t row, int64_t col) const {
    cuda::softmax::Pack<SRC, N> qmk;
    const int64_t offset = (row * row_size + col) / N;
    qmk.storage = *(reinterpret_cast<const cuda::softmax::PackType<SRC, N>*>(q) + offset);
    cuda::softmax::Pack<SRC, N> mask;
    const int64_t m_offset = (row / stride * row_size + col) / N;
    mask.storage = *(reinterpret_cast<const cuda::softmax::PackType<SRC, N>*>(m) + m_offset);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      dst[i] =
          static_cast<DST>(qmk.elem[i]) * static_cast<DST>(scale) + static_cast<DST>(mask.elem[i]);
    }
  }
  const SRC* q;
  const SRC* m;
  const SRC scale;
  int64_t stride;
  int64_t row_size;
};

template<typename T, typename ComputeType = typename cuda::softmax::DefaultComputeType<T>::type>
void LaunchMSAWithBiasSoftmaxForwardKernel(cudaStream_t stream, T* out, const T* qmk, const T* mask,
                                           const T* bias, T scale, const int64_t stride,
                                           const int64_t row_size, const int64_t rows,
                                           const int64_t cols) {
  cuda::softmax::DirectStore<ComputeType, T> store(out, row_size);
  MSALoadWithBias<T, ComputeType> load(qmk, mask, bias, scale, stride, row_size);
  OF_CUDA_CHECK((cuda::softmax::DispatchSoftmax<decltype(load), decltype(store), ComputeType>(
      stream, load, store, rows, cols)));
};

template<typename T, typename ComputeType = typename cuda::softmax::DefaultComputeType<T>::type>
void LaunchMSASoftmaxForwardKernel(cudaStream_t stream, T* out, const T* qmk, const T* mask,
                                   T scale, const int64_t stride, const int64_t row_size,
                                   const int64_t rows, const int64_t cols) {
  cuda::softmax::DirectStore<ComputeType, T> store(out, row_size);
  MSALoad<T, ComputeType> load(qmk, mask, scale, stride, row_size);
  OF_CUDA_CHECK((cuda::softmax::DispatchSoftmax<decltype(load), decltype(store), ComputeType>(
      stream, load, store, rows, cols)));
};

template<typename SRC, typename DST>
struct MSAGradStore {
  MSAGradStore(DST* dx, const SRC scale, int64_t row_size)
      : dx(dx), scale(scale), row_size(row_size) {}
  template<int N>
  __device__ void store(const SRC* dout, int64_t row, int64_t col) const {
    cuda::softmax::Pack<DST, N> qmk;
    const int64_t offset = (row * row_size + col) / N;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      qmk.elem[i] = static_cast<DST>(dout[i]) * static_cast<DST>(scale);
    }
    *(reinterpret_cast<cuda::softmax::PackType<DST, N>*>(dx) + offset) = qmk.storage;
  }
  DST* dx;
  const SRC scale;
  int64_t row_size;
};

template<typename T, typename ComputeType = typename cuda::softmax::DefaultComputeType<T>::type>
void LaunchMSASoftmaxBackwardKernel(cudaStream_t stream, T* dx, const T* y, const T* dy, T scale,
                                    const int64_t row_size, const int64_t rows,
                                    const int64_t cols) {
  MSAGradStore<ComputeType, T> store(dx, scale, row_size);
  cuda::softmax::DirectLoad<T, ComputeType> load_y(y, row_size);
  cuda::softmax::DirectLoad<T, ComputeType> load_dy(dy, row_size);
  OF_CUDA_CHECK((
      cuda::softmax::DispatchSoftmaxGrad<decltype(load_y), decltype(load_dy), decltype(store),
                                         ComputeType>(stream, load_y, load_dy, store, rows, cols)));
};
}  // namespace attn

}  // namespace alphafold

template<typename T>
class FusedMSASoftmaxKernel final : public user_op::OpKernel {
 public:
  FusedMSASoftmaxKernel() = default;
  ~FusedMSASoftmaxKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* qmk = ctx->Tensor4ArgNameAndIndex("qmk", 0);
    const user_op::Tensor* mask = ctx->Tensor4ArgNameAndIndex("mask", 0);
    const T scale = ctx->Attr<float>("scale");
    const std::string mode = ctx->Attr<std::string>("mode");
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    auto qmk_shape = qmk->shape_view();
    auto axes = qmk_shape.NumAxes();
    int64_t B = qmk_shape.At(0), h = qmk_shape.At(1), S1 = qmk_shape.At(2),
            S2 = qmk_shape.At(axes - 1);
    if (mode == "template") {
      B = qmk_shape.At(0) * qmk_shape.At(1);
      h = qmk_shape.At(2);
      S1 = qmk_shape.At(3);
      S2 = qmk_shape.At(4);
    } else if (axes == 5) {
      CHECK_EQ(qmk_shape.At(0), 1);
      B = qmk_shape.At(1);
      h = qmk_shape.At(2);
      S1 = qmk_shape.At(3);
      S2 = qmk_shape.At(4);
    }

    if (ctx->has_input("bias", 0)) {
      const user_op::Tensor* bias = ctx->Tensor4ArgNameAndIndex("bias", 0);
      alphafold::attn::LaunchMSAWithBiasSoftmaxForwardKernel<T>(
          ctx->stream()->As<ep::CudaStream>()->cuda_stream(), out->mut_dptr<T>(), qmk->dptr<T>(),
          mask->dptr<T>(), bias->dptr<T>(), scale, h * S1, S2, B * h * S1, S2);
    } else {
      int64_t stride = mode == "template" ? B * h * S1 : (mode == "col" ? h * S1 : h);
      int64_t rows = mode == "global_col" ? h * B : B * h * S1;
      alphafold::attn::LaunchMSASoftmaxForwardKernel<T>(
          ctx->stream()->As<ep::CudaStream>()->cuda_stream(), out->mut_dptr<T>(), qmk->dptr<T>(),
          mask->dptr<T>(), scale, stride, S2, rows, S2);
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_MSA_SOFTMAX_KERNEL_GPU(dtype)                   \
  REGISTER_USER_KERNEL("fused_msa_softmax")                            \
      .SetCreateFn<FusedMSASoftmaxKernel<dtype>>()                     \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_FUSED_MSA_SOFTMAX_KERNEL_GPU(half)
#if CUDA_VERSION >= 11000
REGISTER_FUSED_MSA_SOFTMAX_KERNEL_GPU(nv_bfloat16)
#endif
REGISTER_FUSED_MSA_SOFTMAX_KERNEL_GPU(float)

template<typename T>
class FusedMSASoftmaxGradKernel final : public user_op::OpKernel {
 public:
  FusedMSASoftmaxGradKernel() = default;
  ~FusedMSASoftmaxGradKernel() override = default;

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

    alphafold::attn::LaunchMSASoftmaxBackwardKernel<T>(
        ctx->stream()->As<ep::CudaStream>()->cuda_stream(), dx->mut_dptr<T>(), y->dptr<T>(),
        dy->dptr<T>(), scale, S, rows, S);
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_MSA_SOFTMAX_GRAD_KERNEL_GPU(dtype)              \
  REGISTER_USER_KERNEL("fused_msa_softmax_grad")                       \
      .SetCreateFn<FusedMSASoftmaxGradKernel<dtype>>()                 \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_FUSED_MSA_SOFTMAX_GRAD_KERNEL_GPU(half)
#if CUDA_VERSION >= 11000
REGISTER_FUSED_MSA_SOFTMAX_GRAD_KERNEL_GPU(nv_bfloat16)
#endif
REGISTER_FUSED_MSA_SOFTMAX_GRAD_KERNEL_GPU(float)

namespace alphafold {
namespace gate {
template<typename T>
__device__ __forceinline__ T Sigmoid(const T x) {
  const T half_of_one = static_cast<T>(0.5);
  return half_of_one * tanh(half_of_one * x) + half_of_one;
}

template<>
__device__ __forceinline__ half Sigmoid(const half x) {
  return __float2half(Sigmoid(__half2float(x)));
}

template<typename T>
__global__ void biasadd_sigmoid_mul_kernel(const int n, const int cols, T* out, const T* b,
                                           const T* g, const T* x) {
  CUDA_1D_KERNEL_LOOP(i, n) { out[i] = x[i] * Sigmoid<T>(g[i] + b[i % cols]); }
}
template<typename T>
__global__ void biasadd_sigmoid_mul_grad_kernel(const int n, const int cols, const T* dout,
                                                const T* b, const T* g, const T* x, T* dg) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    T sigmoid_g = Sigmoid(g[i] + b[i % cols]);
    dg[i] = sigmoid_g * (1 - sigmoid_g) * sigmoid_g * dout[i];
  }
}
}  // namespace gate
};  // namespace alphafold

template<typename T>
void launch_biasadd_sigmoid_mul(cudaStream_t stream, const int n, const int cols, T* out,
                                const T* b, const T* g, const T* x) {
  constexpr int block_dim = 128;
  constexpr int waves = 32;
  const int64_t num_blocks = n / block_dim;
  int grid_dim_x;
  {
    cudaError_t err = cuda::softmax::GetNumBlocks(block_dim, num_blocks, waves, &grid_dim_x);
    OF_CUDA_CHECK(err);
  }
  alphafold::gate::biasadd_sigmoid_mul_kernel<T>
      <<<grid_dim_x, block_dim, 0, stream>>>(n, cols, out, b, g, x);
};

template<typename T>
void launch_biasadd_sigmoid_mul_grad(cudaStream_t stream, const int n, const int cols,
                                     const T* dout, const T* b, const T* g, const T* x, T* dg) {
  constexpr int block_dim = 128;
  constexpr int waves = 32;
  const int64_t num_blocks = n / block_dim;
  int grid_dim_x;
  {
    cudaError_t err = cuda::softmax::GetNumBlocks(block_dim, num_blocks, waves, &grid_dim_x);
    OF_CUDA_CHECK(err);
  }
  alphafold::gate::biasadd_sigmoid_mul_grad_kernel<T>
      <<<grid_dim_x, block_dim, 0, stream>>>(n, cols, dout, b, g, x, dg);
};

template<typename T>
class FusedMSABiasaddSigmoidMulKernel final : public user_op::OpKernel {
 public:
  FusedMSABiasaddSigmoidMulKernel() = default;
  ~FusedMSABiasaddSigmoidMulKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* bias = ctx->Tensor4ArgNameAndIndex("b", 0);
    const user_op::Tensor* gate = ctx->Tensor4ArgNameAndIndex("g", 0);
    auto cnt = gate->shape_view().elem_cnt();
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    CHECK(cnt == x->shape_view().elem_cnt());
    auto cols = bias->shape_view().At(0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    launch_biasadd_sigmoid_mul<T>(ctx->stream()->As<ep::CudaStream>()->cuda_stream(), cnt, cols,
                                  out->mut_dptr<T>(), bias->dptr<T>(), gate->dptr<T>(),
                                  x->dptr<T>());
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_MSA_BIASADD_SIGMOID_MUL_KERNEL_GPU(dtype)       \
  REGISTER_USER_KERNEL("fused_msa_biasadd_sigmoid_mul")                \
      .SetCreateFn<FusedMSABiasaddSigmoidMulKernel<dtype>>()           \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("g", 0) == GetDataType<dtype>::value));

REGISTER_FUSED_MSA_BIASADD_SIGMOID_MUL_KERNEL_GPU(float)

template<typename T>
class FusedMSABiasaddSigmoidMulGradKernel final : public user_op::OpKernel {
 public:
  FusedMSABiasaddSigmoidMulGradKernel() = default;
  ~FusedMSABiasaddSigmoidMulGradKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dout = ctx->Tensor4ArgNameAndIndex("dout", 0);
    const user_op::Tensor* b = ctx->Tensor4ArgNameAndIndex("b", 0);
    const user_op::Tensor* g = ctx->Tensor4ArgNameAndIndex("g", 0);
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    auto cnt = g->shape_view().elem_cnt();
    CHECK(cnt = x->shape_view().elem_cnt());
    auto cols = b->shape_view().At(0);

    user_op::Tensor* dg = ctx->Tensor4ArgNameAndIndex("dg", 0);

    CHECK(cnt == dg->shape_view().elem_cnt());
    launch_biasadd_sigmoid_mul_grad(ctx->stream()->As<ep::CudaStream>()->cuda_stream(), cnt, cols,
                                    dout->dptr<T>(), b->dptr<T>(), g->dptr<T>(), x->dptr<T>(),
                                    dg->mut_dptr<T>());
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_MSA_BIASADD_SIGMOID_MUL_GRAD_KERNEL_GPU(dtype)  \
  REGISTER_USER_KERNEL("fused_msa_biasadd_sigmoid_mul_grad")           \
      .SetCreateFn<FusedMSABiasaddSigmoidMulGradKernel<dtype>>()       \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("dout", 0) == GetDataType<dtype>::value));

REGISTER_FUSED_MSA_BIASADD_SIGMOID_MUL_GRAD_KERNEL_GPU(float)

namespace alphafold {
namespace dropout_add {

template<typename T>
__global__ void biasadd_dropout_residual_kernel(const int n, const int b_stride,
                                                const int mask_stride, T* out, const T* b,
                                                const T* x, const T* mask, const T* residual) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    out[i] = (x[i] + b[i % b_stride]) * mask[i % mask_stride] + residual[i];
  }
}
template<typename T>
__global__ void biasadd_dropout_residual_grad_kernel(const int n, const int mask_stride,
                                                     const T* dout, const T* mask, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, n) { dx[i] = dout[i] * mask[i % mask_stride]; }
}

}  // namespace dropout_add
}  // namespace alphafold
template<typename T>
void launch_biasadd_dropout_residual(cudaStream_t stream, const int n, const int b_stride,
                                     const int mask_stride, T* out, const T* bias, const T* x,
                                     const T* mask, const T* residual) {
  constexpr int block_dim = 128;
  constexpr int waves = 32;
  const int64_t num_blocks = n / block_dim;
  int grid_dim_x;
  {
    cudaError_t err = cuda::softmax::GetNumBlocks(block_dim, num_blocks, waves, &grid_dim_x);
    OF_CUDA_CHECK(err);
  }
  alphafold::dropout_add::biasadd_dropout_residual_kernel<T><<<grid_dim_x, block_dim, 0, stream>>>(
      n, b_stride, mask_stride, out, bias, x, mask, residual);
};

template<typename T>
void launch_biasadd_dropout_residual_grad(cudaStream_t stream, const int n, const int mask_stride,
                                          const T* dout, const T* mask, T* dx) {
  constexpr int block_dim = 128;
  constexpr int waves = 32;
  const int64_t num_blocks = n / block_dim;
  int grid_dim_x;
  {
    cudaError_t err = cuda::softmax::GetNumBlocks(block_dim, num_blocks, waves, &grid_dim_x);
    OF_CUDA_CHECK(err);
  }
  alphafold::dropout_add::biasadd_dropout_residual_grad_kernel<T>
      <<<grid_dim_x, block_dim, 0, stream>>>(n, mask_stride, dout, mask, dx);
};

template<typename T>
class FusedMSABiasaddDropoutResidualKernel final : public user_op::OpKernel {
 public:
  FusedMSABiasaddDropoutResidualKernel() = default;
  ~FusedMSABiasaddDropoutResidualKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* bias = ctx->Tensor4ArgNameAndIndex("bias", 0);  // broadcast
    const user_op::Tensor* mask = ctx->Tensor4ArgNameAndIndex("mask", 0);  // broadcast
    const user_op::Tensor* res = ctx->Tensor4ArgNameAndIndex("residual", 0);
    auto cnt = res->shape_view().elem_cnt();
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    CHECK(cnt == x->shape_view().elem_cnt() && cnt == out->shape_view().elem_cnt());
    auto b_stride = bias->shape_view().At(0);
    auto axes = mask->shape_view().NumAxes();
    auto mask_stride = mask->shape_view().At(axes - 2) * b_stride;

    launch_biasadd_dropout_residual(ctx->stream()->As<ep::CudaStream>()->cuda_stream(), cnt,
                                    b_stride, mask_stride, out->mut_dptr<T>(), bias->dptr<T>(),
                                    x->dptr<T>(), mask->dptr<T>(), res->dptr<T>());
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_MSA_BIASADD_DROPOUT_RESIDUAL_KERNEL_GPU(dtype)  \
  REGISTER_USER_KERNEL("fused_msa_biasadd_dropout_residual")           \
      .SetCreateFn<FusedMSABiasaddDropoutResidualKernel<dtype>>()      \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_FUSED_MSA_BIASADD_DROPOUT_RESIDUAL_KERNEL_GPU(float)

template<typename T>
class FusedMSABiasaddDropoutResidualGradKernel final : public user_op::OpKernel {
 public:
  FusedMSABiasaddDropoutResidualGradKernel() = default;
  ~FusedMSABiasaddDropoutResidualGradKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dout = ctx->Tensor4ArgNameAndIndex("dout", 0);  // broadcast
    const user_op::Tensor* mask = ctx->Tensor4ArgNameAndIndex("mask", 0);
    auto cnt = dout->shape_view().elem_cnt();
    CHECK(cnt == mask->shape_view().elem_cnt());
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    CHECK(cnt == dx->shape_view().elem_cnt());
    auto axes = mask->shape_view().NumAxes();
    auto mask_stride = mask->shape_view().At(axes - 2) * mask->shape_view().At(axes - 1);

    launch_biasadd_dropout_residual_grad(ctx->stream()->As<ep::CudaStream>()->cuda_stream(), cnt,
                                         mask_stride, dout->dptr<T>(), mask->dptr<T>(),
                                         dx->mut_dptr<T>());
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_MSA_BIASADD_DROPOUT_RESIDUAL_GRAD_KERNEL_GPU(dtype) \
  REGISTER_USER_KERNEL("fused_msa_biasadd_dropout_residual_grad")          \
      .SetCreateFn<FusedMSABiasaddDropoutResidualGradKernel<dtype>>()      \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)     \
                       && (user_op::HobDataType("dout", 0) == GetDataType<dtype>::value));

REGISTER_FUSED_MSA_BIASADD_DROPOUT_RESIDUAL_GRAD_KERNEL_GPU(float)

namespace alphafold {
namespace tmu {  // triangular multiplicative update
template<typename T>
__global__ void tri_mul_update_kernel(const int n, const int b1_stride, const int b2_stride,
                                      const int mask_stride, T* out, const T* x1, const T* b1,
                                      const T* x2, const T* b2, const T* mask, const T* residual) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    out[i] = (x1[i] + b1[i % b1_stride]) * alphafold::gate::Sigmoid(x2[i] + b2[i % b2_stride])
                 * mask[i % mask_stride]
             + residual[i];
  }
}

template<typename T>
__global__ void tri_mul_update_grad_kernel(const int n, const int b1_stride, const int b2_stride,
                                           const int mask_stride, const T* dout, const T* x1,
                                           const T* b1, const T* x2, const T* b2, const T* mask,
                                           T* dx1, T* db1, T* dx2, T* db2, T* dr) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    auto sigmoid_o = alphafold::gate::Sigmoid(x2[i] + b2[i % b2_stride]);
    auto out1 = x1[i] + b1[i % b1_stride];
    dx1[i] = dout[i] * sigmoid_o * mask[i % mask_stride];
    db1[i] = dx1[i];
    dx2[i] = (x1[i] + b1[i % b1_stride]) * mask[i % mask_stride] * sigmoid_o * (1 - sigmoid_o);
    db2[i] = dx2[i];
    dr[i] = dout[i];
  }
}

}  // namespace tmu
}  // namespace alphafold
template<typename T>
void launch_tri_mul_update(cudaStream_t stream, const int n, const int b1_stride,
                           const int b2_stride, const int mask_stride, T* out, const T* x1,
                           const T* b1, const T* x2, const T* b2, const T* mask,
                           const T* residual) {
  constexpr int block_dim = 128;
  constexpr int waves = 32;
  const int64_t num_blocks = n / block_dim;
  int grid_dim_x;
  {
    cudaError_t err = cuda::softmax::GetNumBlocks(block_dim, num_blocks, waves, &grid_dim_x);
    OF_CUDA_CHECK(err);
  }
  alphafold::tmu::tri_mul_update_kernel<T><<<grid_dim_x, block_dim, 0, stream>>>(
      n, b1_stride, b2_stride, mask_stride, out, x1, b1, x2, b2, mask, residual);
};

template<typename T>
void launch_tri_mul_update_grad(cudaStream_t stream, const int n, const int b1_stride,
                                const int b2_stride, const int mask_stride, const T* dout,
                                const T* x1, const T* b1, const T* x2, const T* b2, const T* mask,
                                T* dx1, T* db1, T* dx2, T* db2, T* dr) {
  constexpr int block_dim = 128;
  constexpr int waves = 32;
  const int64_t num_blocks = n / block_dim;
  int grid_dim_x;
  {
    cudaError_t err = cuda::softmax::GetNumBlocks(block_dim, num_blocks, waves, &grid_dim_x);
    OF_CUDA_CHECK(err);
  }
  alphafold::tmu::tri_mul_update_grad_kernel<T><<<grid_dim_x, block_dim, 0, stream>>>(
      n, b1_stride, b2_stride, mask_stride, dout, x1, b1, x2, b2, mask, dx1, db1, dx2, db2, dr);
};

template<typename T>
class FusedMSATriMulUpdateKernel final : public user_op::OpKernel {
 public:
  FusedMSATriMulUpdateKernel() = default;
  ~FusedMSATriMulUpdateKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x1 = ctx->Tensor4ArgNameAndIndex("x1", 0);      // broadcast
    const user_op::Tensor* b1 = ctx->Tensor4ArgNameAndIndex("b1", 0);      // broadcast
    const user_op::Tensor* x2 = ctx->Tensor4ArgNameAndIndex("x2", 0);      // broadcast
    const user_op::Tensor* b2 = ctx->Tensor4ArgNameAndIndex("b2", 0);      // broadcast
    const user_op::Tensor* mask = ctx->Tensor4ArgNameAndIndex("mask", 0);  // broadcast
    const user_op::Tensor* res = ctx->Tensor4ArgNameAndIndex("residual", 0);
    auto cnt = res->shape_view().elem_cnt();
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    CHECK(cnt == x1->shape_view().elem_cnt() && cnt == out->shape_view().elem_cnt());
    auto b1_stride = b1->shape_view().At(0);
    auto b2_stride = b2->shape_view().At(0);
    auto axes = mask->shape_view().NumAxes();
    auto mask_stride = mask->shape_view().At(axes - 2) * mask->shape_view().At(axes - 1);

    launch_tri_mul_update(ctx->stream()->As<ep::CudaStream>()->cuda_stream(), cnt, b1_stride,
                          b2_stride, mask_stride, out->mut_dptr<T>(), x1->dptr<T>(), b1->dptr<T>(),
                          x2->dptr<T>(), b2->dptr<T>(), mask->dptr<T>(), res->dptr<T>());
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_MSA_TRI_MUL_UPDATE_KERNEL_GPU(dtype)            \
  REGISTER_USER_KERNEL("fused_msa_tmu")                                \
      .SetCreateFn<FusedMSATriMulUpdateKernel<dtype>>()                \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_FUSED_MSA_TRI_MUL_UPDATE_KERNEL_GPU(float)

template<typename T>
class FusedMSATriMulUpdateGradKernel final : public user_op::OpKernel {
 public:
  FusedMSATriMulUpdateGradKernel() = default;
  ~FusedMSATriMulUpdateGradKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dout = ctx->Tensor4ArgNameAndIndex("dout", 0);  // broadcast
    const user_op::Tensor* mask = ctx->Tensor4ArgNameAndIndex("mask", 0);
    const user_op::Tensor* x1 = ctx->Tensor4ArgNameAndIndex("x1", 0);
    const user_op::Tensor* b1 = ctx->Tensor4ArgNameAndIndex("b1", 0);
    const user_op::Tensor* x2 = ctx->Tensor4ArgNameAndIndex("x2", 0);
    const user_op::Tensor* b2 = ctx->Tensor4ArgNameAndIndex("b2", 0);
    auto cnt = dout->shape_view().elem_cnt();
    CHECK(cnt == mask->shape_view().elem_cnt());
    user_op::Tensor* dx1 = ctx->Tensor4ArgNameAndIndex("dx1", 0);
    user_op::Tensor* db1 = ctx->Tensor4ArgNameAndIndex("db1", 0);
    user_op::Tensor* dx2 = ctx->Tensor4ArgNameAndIndex("dx2", 0);
    user_op::Tensor* db2 = ctx->Tensor4ArgNameAndIndex("db2", 0);
    user_op::Tensor* dr = ctx->Tensor4ArgNameAndIndex("dr", 0);
    CHECK(cnt == dx1->shape_view().elem_cnt());
    auto axes = dx1->shape_view().NumAxes();
    auto b1_stride = dx1->shape_view().At(axes - 1);
    auto b2_stride = dx2->shape_view().At(axes - 1);
    auto axes_m = mask->shape_view().NumAxes();
    auto mask_stride = mask->shape_view().At(axes_m - 2) * mask->shape_view().At(axes_m - 1);

    launch_tri_mul_update_grad(ctx->stream()->As<ep::CudaStream>()->cuda_stream(), cnt, b1_stride,
                               b2_stride, mask_stride, dout->dptr<T>(), x1->dptr<T>(),
                               b1->dptr<T>(), x2->dptr<T>(), b2->dptr<T>(), mask->dptr<T>(),
                               dx1->mut_dptr<T>(), db1->mut_dptr<T>(), dx2->mut_dptr<T>(),
                               db2->mut_dptr<T>(), dr->mut_dptr<T>());
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_MSA_TRI_MUL_UPDATE_GRAD_KERNEL_GPU(dtype)       \
  REGISTER_USER_KERNEL("fused_msa_tmu_grad")                           \
      .SetCreateFn<FusedMSATriMulUpdateGradKernel<dtype>>()            \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("dout", 0) == GetDataType<dtype>::value));

REGISTER_FUSED_MSA_TRI_MUL_UPDATE_GRAD_KERNEL_GPU(float)
}  // namespace oneflow
