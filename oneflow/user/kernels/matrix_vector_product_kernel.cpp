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
#include "oneflow/core/ep/include/primitive/memcpy.h"
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
std::unique_ptr<ep::primitive::Matmul> NewMatrixVectorProductPrimitive(Context* ctx) {
  const DataType data_type = ctx->TensorDesc4ArgNameAndIndex("out", 0)->data_type();
  return NewMatmulPrimitive(ctx->device_type(), data_type, false, false);
}

template<typename Context>
std::unique_ptr<ep::primitive::Matmul> NewMatrixVectorProductGradAPrimitive(Context* ctx) {
  const DataType data_type = ctx->TensorDesc4ArgNameAndIndex("dx", 0)->data_type();
  return NewMatmulPrimitive(ctx->device_type(), data_type, false, true);
}

template<typename Context>
std::unique_ptr<ep::primitive::Matmul> NewMatrixVectorProductGradBPrimitive(Context* ctx) {
  const DataType data_type = ctx->TensorDesc4ArgNameAndIndex("dx", 0)->data_type();
  return NewMatmulPrimitive(ctx->device_type(), data_type, true, false);
}

auto MatrixVectorProductPrimitiveExists() {
  return hob::make_custom("NewMatrixVectorProductPrimitiveExists",
                          [](const user_op::KernelRegContext& ctx) {
                            return NewMatrixVectorProductPrimitive(&ctx).operator bool();
                          });
}

auto MatrixVectorProductGradAPrimitiveExists() {
  return hob::make_custom("NewMatrixVectorProductGradAPrimitiveExists",
                          [](const user_op::KernelRegContext& ctx) {
                            return NewMatrixVectorProductGradAPrimitive(&ctx).operator bool();
                          });
}

auto MatrixVectorProductGradBPrimitiveExists() {
  return hob::make_custom("NewMatrixVectorProductGradBPrimitiveExists",
                          [](const user_op::KernelRegContext& ctx) {
                            return NewMatrixVectorProductGradBPrimitive(&ctx).operator bool();
                          });
}

class MatrixVectorProductKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  MatrixVectorProductKernel() = default;
  ~MatrixVectorProductKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    /*
    A(m, k) matmul B(k) -> (m, k) matmul (k, 1) -> (m, 1) -> (m)
    */
    const user_op::Tensor* a = ctx->Tensor4ArgNameAndIndex("a", 0);
    CHECK_EQ(a->shape_view().NumAxes(), 2) << "A Numdims should be equal to 2. ";
    const DataType data_type = a->data_type();
    const user_op::Tensor* b = ctx->Tensor4ArgNameAndIndex("b", 0);
    CHECK_EQ(b->shape_view().NumAxes(), 1) << "B Numdims should be equal to 1. ";
    CHECK_EQ(b->data_type(), data_type) << "Matrix A Datatype should be equal to Vector B";
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    CHECK_EQ(out->shape_view().NumAxes(), 1) << "Out Numdims should be equal to 1. ";
    CHECK_EQ(out->data_type(), data_type) << "Out Datatype should be equal to input's. ";
    size_t m = a->shape_view().At(0);
    size_t k = a->shape_view().At(1);
    size_t n = 1;
    const double alpha = 1.0;
    double beta = 0.0;
    auto matmul = NewMatrixVectorProductPrimitive(ctx);
    CHECK(matmul);
    matmul->Launch(ctx->stream(), m, n, k, alpha, a->dptr(), b->dptr(), beta, out->mut_dptr());
  }
};

REGISTER_USER_KERNEL("matrix_vector_product")
    .SetCreateFn<MatrixVectorProductKernel>()
    .SetIsMatchedHob(MatrixVectorProductPrimitiveExists());

class MatrixVectorProductGradAKernel final : public user_op::OpKernel,
                                             public user_op::CudaGraphSupport {
 public:
  MatrixVectorProductGradAKernel() = default;
  ~MatrixVectorProductGradAKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    /*
    A(m, k) matmul B(k) -> (m, k) matmul (k, 1) -> (m, 1) -> (m)
    GradA = dy (m) matmul B(k) -> (m, 1) (k, 1)_transpose
    */
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* b = ctx->Tensor4ArgNameAndIndex("b", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    size_t m = dy->shape_view().At(0);
    size_t k = 1;
    size_t n = b->shape_view().At(0);
    const double alpha = 1.0;
    double beta = 0.0;
    auto matmul = NewMatrixVectorProductGradAPrimitive(ctx);
    CHECK(matmul);
    matmul->Launch(ctx->stream(), m, n, k, alpha, dy->dptr(), b->dptr(), beta, dx->mut_dptr());
  }
};

REGISTER_USER_KERNEL("matrix_vector_product_grad_a")
    .SetCreateFn<MatrixVectorProductGradAKernel>()
    .SetIsMatchedHob(MatrixVectorProductGradAPrimitiveExists());

class MatrixVectorProductGradBKernel final : public user_op::OpKernel,
                                             public user_op::CudaGraphSupport {
 public:
  MatrixVectorProductGradBKernel() = default;
  ~MatrixVectorProductGradBKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    /*
    A(m, k) matmul B(k) -> (m, k) matmul (k, 1) -> (m, 1) -> (m)
    GradB = dy_transpose (1, m) matmul A(m, k)
    */
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* a = ctx->Tensor4ArgNameAndIndex("a", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    size_t m = 1;
    size_t k = dy->shape_view().At(0);
    size_t n = a->shape_view().At(1);
    const double alpha = 1.0;
    double beta = 0.0;
    auto matmul = NewMatrixVectorProductGradBPrimitive(ctx);
    CHECK(matmul);
    matmul->Launch(ctx->stream(), m, n, k, alpha, dy->dptr(), a->dptr(), beta, dx->mut_dptr());
  }
};

REGISTER_USER_KERNEL("matrix_vector_product_grad_b")
    .SetCreateFn<MatrixVectorProductGradBKernel>()
    .SetIsMatchedHob(MatrixVectorProductGradBPrimitiveExists());

}  // namespace

}  // namespace oneflow
