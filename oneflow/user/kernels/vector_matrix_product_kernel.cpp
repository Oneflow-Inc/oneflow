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

ep::primitive::BlasTransposeType GetBlasTransposeType(bool transpose) {
  return transpose ? ep::primitive::BlasTransposeType::T : ep::primitive::BlasTransposeType::N;
}

template<typename Context>
std::unique_ptr<ep::primitive::Memcpy> NewMemcpyPrimitive(Context* ctx) {
  return ep::primitive::NewPrimitive<ep::primitive::MemcpyFactory>(
      ctx->device_type(), ep::primitive::MemcpyKind::kDtoD);
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
std::unique_ptr<ep::primitive::Matmul> NewVectorMatrixProductPrimitive(Context* ctx) {
  const DataType data_type = ctx->TensorDesc4ArgNameAndIndex("out", 0)->data_type();
  return NewMatmulPrimitive(ctx->device_type(), data_type, false, false);
}

template<typename Context>
std::unique_ptr<ep::primitive::Matmul> NewVectorMatrixProductGradAPrimitive(Context* ctx) {
  const DataType data_type = ctx->TensorDesc4ArgNameAndIndex("dx", 0)->data_type();
  return NewMatmulPrimitive(ctx->device_type(), data_type, false, true);
}

template<typename Context>
std::unique_ptr<ep::primitive::Matmul> NewVectorMatrixProductGradBPrimitive(Context* ctx) {
  const DataType data_type = ctx->TensorDesc4ArgNameAndIndex("dx", 0)->data_type();
  return NewMatmulPrimitive(ctx->device_type(), data_type, true, false);
}

auto VectorMatrixProductPrimitiveExists() {
  return hob::make_custom("NewVectorMatrixProductPrimitiveExists",
                          [](const user_op::KernelRegContext& ctx) {
                            return NewVectorMatrixProductPrimitive(&ctx).operator bool();
                          });
}

auto VectorMatrixProductGradAPrimitiveExists() {
  return hob::make_custom("NewVectorMatrixProductGradAPrimitiveExists",
                          [](const user_op::KernelRegContext& ctx) {
                            return NewVectorMatrixProductGradAPrimitive(&ctx).operator bool();
                          });
}

auto VectorMatrixProductGradBPrimitiveExists() {
  return hob::make_custom("NewVectorMatrixProductGradBPrimitiveExists",
                          [](const user_op::KernelRegContext& ctx) {
                            return NewVectorMatrixProductGradBPrimitive(&ctx).operator bool();
                          });
}

class VectorMatrixProductKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  VectorMatrixProductKernel() = default;
  ~VectorMatrixProductKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    /*
    A(k, ) matmul B(k, n) -> (1, k) matmul (k, n) -> (1, n) -> (n)
    */
    const user_op::Tensor* a = ctx->Tensor4ArgNameAndIndex("a", 0);
    CHECK_EQ(a->shape_view().NumAxes(), 1) << "A Numdims should be equal to 1. ";
    const DataType data_type = a->data_type();
    const user_op::Tensor* b = ctx->Tensor4ArgNameAndIndex("b", 0);
    CHECK_EQ(b->shape_view().NumAxes(), 2) << "B Numdims should be equal to 2. ";
    CHECK_EQ(b->data_type(), data_type) << "Matrix A Datatype should be equal to Vector B";
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    CHECK_EQ(out->shape_view().NumAxes(), 1) << "Out Numdims should be equal to 1. ";
    CHECK_EQ(out->data_type(), data_type) << "Out Datatype should be equal to input's. ";
    size_t m = 1;
    size_t k = a->shape_view().At(0);
    size_t n = b->shape_view().At(1);
    const double alpha = 1.0;
    double beta = 0.0;
    auto matmul = NewVectorMatrixProductPrimitive(ctx);
    CHECK(matmul);
    matmul->Launch(ctx->stream(), m, n, k, alpha, a->dptr(), b->dptr(), beta, out->mut_dptr());
  }
};

REGISTER_USER_KERNEL("vector_matrix_product")
    .SetCreateFn<VectorMatrixProductKernel>()
    .SetIsMatchedHob(VectorMatrixProductPrimitiveExists());

class VectorMatrixProductGradAKernel final : public user_op::OpKernel,
                                             public user_op::CudaGraphSupport {
 public:
  VectorMatrixProductGradAKernel() = default;
  ~VectorMatrixProductGradAKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    /*
    A(k, ) matmul B(k, n) -> (1, k) matmul (k, n) -> (1, n) -> (n)
    GradA = dy (n) matmul B_transpose(n, k) -> (1, n) matmul (n, k)
    */
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* b = ctx->Tensor4ArgNameAndIndex("b", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    size_t m = 1;
    size_t k = dy->shape_view().At(0);
    size_t n = b->shape_view().At(0);
    const double alpha = 1.0;
    double beta = 0.0;
    auto matmul = NewVectorMatrixProductGradAPrimitive(ctx);
    CHECK(matmul);
    matmul->Launch(ctx->stream(), m, n, k, alpha, dy->dptr(), b->dptr(), beta, dx->mut_dptr());
  }
};

REGISTER_USER_KERNEL("vector_matrix_product_grad_a")
    .SetCreateFn<VectorMatrixProductGradAKernel>()
    .SetIsMatchedHob(VectorMatrixProductGradAPrimitiveExists());

class VectorMatrixProductGradBKernel final : public user_op::OpKernel,
                                             public user_op::CudaGraphSupport {
 public:
  VectorMatrixProductGradBKernel() = default;
  ~VectorMatrixProductGradBKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    /*
    A(k, ) matmul B(k, n) -> (1, k) matmul (k, n) -> (1, n) -> (n)
    GradB = a_transpose (k, 1) matmul dy (1, n)
    */
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* a = ctx->Tensor4ArgNameAndIndex("a", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    size_t m = a->shape_view().At(0);
    size_t k = 1;
    size_t n = dy->shape_view().At(0);
    const double alpha = 1.0;
    double beta = 0.0;
    auto matmul = NewVectorMatrixProductGradBPrimitive(ctx);
    CHECK(matmul);
    matmul->Launch(ctx->stream(), m, n, k, alpha, a->dptr(), dy->dptr(), beta, dx->mut_dptr());
  }
};

REGISTER_USER_KERNEL("vector_matrix_product_grad_b")
    .SetCreateFn<VectorMatrixProductGradBKernel>()
    .SetIsMatchedHob(VectorMatrixProductGradBPrimitiveExists());

}  // namespace

}  // namespace oneflow
