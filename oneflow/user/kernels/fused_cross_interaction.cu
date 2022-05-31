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
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/framework/config_def.h"
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/ep/include/primitive/matmul.h"

namespace oneflow {

namespace {

void InferMatmulMNK(const ShapeView& a_shape, const ShapeView& b_shape, bool transpose_a,
                    bool transpose_b, size_t* m, size_t* n, size_t* k) {
  const int64_t num_a_axes = a_shape.NumAxes();
  CHECK_GE(num_a_axes, 2);
  const int64_t num_b_axes = b_shape.NumAxes();
  CHECK_GE(num_b_axes, 2);
  if (!transpose_a) {
    *m = a_shape.At(num_a_axes - 2);
    *k = a_shape.At(num_a_axes - 1);
  } else {
    *m = a_shape.At(num_a_axes - 1);
    *k = a_shape.At(num_a_axes - 2);
  }
  if (!transpose_b) {
    CHECK_EQ(b_shape.At(num_b_axes - 2), *k);
    *n = b_shape.At(num_b_axes - 1);
  } else {
    CHECK_EQ(b_shape.At(num_b_axes - 1), *k);
    *n = b_shape.At(num_b_axes - 2);
  }
}

ep::primitive::BlasTransposeType GetBlasTransposeType(bool transpose) {
  return transpose ? ep::primitive::BlasTransposeType::T : ep::primitive::BlasTransposeType::N;
}

std::unique_ptr<ep::primitive::Matmul> NewMatmulPrimitive(DeviceType device_type,
                                                          DataType data_type, bool transpose_a,
                                                          bool transpose_b) {
  const auto trans_a = GetBlasTransposeType(transpose_a);
  const auto trans_b = GetBlasTransposeType(transpose_b);
  return ep::primitive::NewPrimitive<ep::primitive::MatmulFactory>(device_type, data_type, trans_a,
                                                                   trans_b);
}

template<typename Context>
std::unique_ptr<ep::primitive::Matmul> NewMatmulPrimitive(Context* ctx) {
  const DataType data_type = ctx->TensorDesc4ArgNameAndIndex("x", 0)->data_type();
  return NewMatmulPrimitive(ctx->device_type(), data_type, /*transpose_a=*/false,
                            /*transpose_b=*/true);
}

auto MatmulPrimitiveExists() {
  return hob::make_custom("MatmulPrimitiveExists", [](const user_op::KernelRegContext& ctx) {
    return NewMatmulPrimitive(&ctx).operator bool();
  });
}

template<typename T>
__global__ void FusedBroadcastMulAdd(const T* in, const T* x0, const T* bias, T* out, int64_t cols,
                                     int64_t elem_cnt) {
  /*
  in: batch, 1
  x0: batch, hidden,
  bias: hidden
  */
  // TODO: pack
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const int64_t row = i / cols;
    const int64_t col = i - row * cols;
    out[i] = in[row] * x0[i] + bias[col];
  }
}

template<typename T>
class FusedCrossInteractionKernel final : public user_op::OpKernel,
                                          public user_op::CudaGraphSupport {
 public:
  FusedCrossInteractionKernel() = default;
  ~FusedCrossInteractionKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
    const user_op::Tensor* x_0 = ctx->Tensor4ArgNameAndIndex("x_0", 0);
    const user_op::Tensor* bias = ctx->Tensor4ArgNameAndIndex("bias", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* matmul_out_buf = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);

    CHECK_EQ(out->shape().NumAxes(), 2);
    size_t m = 0, n = 0, k = 0;
    InferMatmulMNK(x->shape(), weight->shape(), /*trans_a=*/false, /*trans_b=*/true, &m, &n, &k);
    const double alpha = 1.0;
    double beta = 0.0;
    auto matmul = NewMatmulPrimitive(ctx);
    CHECK(matmul);
    matmul->Launch(ctx->stream(), m, n, k, alpha, x->dptr(), weight->dptr(), beta,
                   matmul_out_buf->mut_dptr());
    OF_CUDA_CHECK(cudaDeviceSynchronize());
    OF_CUDA_CHECK(cudaGetLastError());
    const int64_t elem_cnt = out->shape().elem_cnt();
    const int64_t grid_size = (elem_cnt + 256 - 1) / 256 * 256;
    const int64_t cols = out->shape().At(1);
    printf("Bias is %ld \n", bias->shape().At(0));
    printf("x0 row is %ld , col is %ld \n", out->shape().At(0), out->shape().At(1));
    printf("Output row is %ld , col is %ld \n", x_0->shape().At(0), x_0->shape().At(1));
    FusedBroadcastMulAdd<T>
        <<<grid_size, 256, 0, ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
            reinterpret_cast<T*>(matmul_out_buf->mut_dptr()), x_0->dptr<T>(), bias->dptr<T>(),
            out->mut_dptr<T>(), cols, elem_cnt);
    OF_CUDA_CHECK(cudaDeviceSynchronize());
    OF_CUDA_CHECK(cudaGetLastError());
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_CROSS_INTERACTION_KERNEL(dtype)                                \
  REGISTER_USER_KERNEL("fused_cross_interaction")                                     \
      .SetCreateFn<FusedCrossInteractionKernel<dtype>>()                              \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value) \
                       && MatmulPrimitiveExists())                                    \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                             \
        size_t tmp_size = 0;                                                          \
        const user_op::TensorDesc& x = ctx->InputTensorDesc("x", 0);                  \
        const user_op::TensorDesc& weight = ctx->InputTensorDesc("weight", 0);        \
        const int64_t matmul_out_elem_cnt = x.shape().At(0) * weight.shape().At(0);   \
        tmp_size = GetCudaAlignedSize(matmul_out_elem_cnt * sizeof(dtype));           \
        return tmp_size;                                                              \
      });

REGISTER_FUSED_CROSS_INTERACTION_KERNEL(float)

}  // namespace

}  // namespace oneflow
