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
#include "oneflow/core/ep/include/primitive/batch_matmul.h"
#include "oneflow/core/ep/include/primitive/broadcast_matmul.h"

namespace oneflow {

namespace {

ep::primitive::BlasTransposeType GetBlasTransposeType(bool transpose) {
  return transpose ? ep::primitive::BlasTransposeType::T : ep::primitive::BlasTransposeType::N;
}

template<typename Context>
ep::primitive::BlasTransposeType GetBlasTransposeType(Context* ctx, const std::string& attr) {
  return GetBlasTransposeType(ctx->template Attr<bool>(attr));
}

void InferMatmulMNK(const ShapeView& a_shape, const ShapeView& b_shape, const ShapeView& c_shape,
                    ep::primitive::BlasTransposeType transpose_a,
                    ep::primitive::BlasTransposeType transpose_b, size_t* m, size_t* n, size_t* k) {
  const int64_t num_a_axes = a_shape.NumAxes();
  CHECK_GE(num_a_axes, 2);
  const int64_t num_b_axes = b_shape.NumAxes();
  CHECK_GE(num_b_axes, 2);
  const int64_t num_c_axes = c_shape.NumAxes();
  CHECK_GE(num_c_axes, 2);
  if (transpose_a == ep::primitive::BlasTransposeType::N) {
    *m = a_shape.At(num_a_axes - 2);
    *k = a_shape.At(num_a_axes - 1);
  } else if (transpose_a == ep::primitive::BlasTransposeType::T) {
    *m = a_shape.At(num_a_axes - 1);
    *k = a_shape.At(num_a_axes - 2);
  } else {
    UNIMPLEMENTED();
  }
  if (transpose_b == ep::primitive::BlasTransposeType::N) {
    CHECK_EQ(b_shape.At(num_b_axes - 2), *k);
    *n = b_shape.At(num_b_axes - 1);
  } else if (transpose_b == ep::primitive::BlasTransposeType::T) {
    CHECK_EQ(b_shape.At(num_b_axes - 1), *k);
    *n = b_shape.At(num_b_axes - 2);
  } else {
    UNIMPLEMENTED();
  }
  CHECK_EQ(c_shape.At(num_c_axes - 2), *m);
  CHECK_EQ(c_shape.At(num_c_axes - 1), *n);
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
std::unique_ptr<ep::primitive::Matmul> NewMatmulPrimitive(Context* ctx) {
  const DataType data_type = ctx->TensorDesc4ArgNameAndIndex("out", 0)->data_type();
  return NewMatmulPrimitive(ctx->device_type(), data_type, ctx->template Attr<bool>("transpose_a"),
                            ctx->template Attr<bool>("transpose_b"));
}

template<typename Context>
std::unique_ptr<ep::primitive::BatchMatmul> NewBatchMatmulPrimitive(Context* ctx) {
  const DataType data_type = ctx->TensorDesc4ArgNameAndIndex("out", 0)->data_type();
  const auto trans_a = GetBlasTransposeType(ctx, "transpose_a");
  const auto trans_b = GetBlasTransposeType(ctx, "transpose_b");
  return ep::primitive::NewPrimitive<ep::primitive::BatchMatmulFactory>(
      ctx->device_type(), data_type, trans_a, trans_b);
}

template<typename Context>
std::unique_ptr<ep::primitive::BroadcastMatmul> NewBroadcastMatmulPrimitive(Context* ctx) {
  const DataType data_type = ctx->TensorDesc4ArgNameAndIndex("out", 0)->data_type();
  const auto trans_a = GetBlasTransposeType(ctx, "transpose_a");
  const auto trans_b = GetBlasTransposeType(ctx, "transpose_b");
  const int64_t a_num_axes = ctx->TensorDesc4ArgNameAndIndex("a", 0)->shape().NumAxes();
  const int64_t b_num_axes = ctx->TensorDesc4ArgNameAndIndex("b", 0)->shape().NumAxes();
  const int64_t max_num_axes = std::max(a_num_axes, b_num_axes);
  return ep::primitive::NewPrimitive<ep::primitive::BroadcastMatmulFactory>(
      ctx->device_type(), data_type, trans_a, trans_b, max_num_axes);
}

auto MemcpyPrimitiveExists() {
  return hob::make_custom("MemcpyPrimitiveExists", [](const user_op::KernelRegContext& ctx) {
    return NewMemcpyPrimitive(&ctx).operator bool();
  });
}

auto MatmulPrimitiveExists() {
  return hob::make_custom("MatmulPrimitiveExists", [](const user_op::KernelRegContext& ctx) {
    return NewMatmulPrimitive(&ctx).operator bool();
  });
}

auto BatchMatmulPrimitiveExists() {
  return hob::make_custom("BatchMatmulPrimitiveExists", [](const user_op::KernelRegContext& ctx) {
    return NewBatchMatmulPrimitive(&ctx).operator bool();
  });
}

auto BroadcastMatmulPrimitiveExists() {
  return hob::make_custom("BroadcastMatmulPrimitiveExists",
                          [](const user_op::KernelRegContext& ctx) {
                            return NewBroadcastMatmulPrimitive(&ctx).operator bool();
                          });
}

class MatmulKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  MatmulKernel() = default;
  ~MatmulKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto trans_a = GetBlasTransposeType(ctx, "transpose_a");
    const auto trans_b = GetBlasTransposeType(ctx, "transpose_b");
    const user_op::Tensor* a = ctx->Tensor4ArgNameAndIndex("a", 0);
    CHECK_EQ(a->shape_view().NumAxes(), 2);
    const DataType data_type = a->data_type();
    const user_op::Tensor* b = ctx->Tensor4ArgNameAndIndex("b", 0);
    CHECK_EQ(b->shape_view().NumAxes(), 2);
    CHECK_EQ(b->data_type(), data_type);

    const int32_t elem_cnt_a = a->shape_view().elem_cnt();
    const int32_t elem_cnt_b = b->shape_view().elem_cnt();
    CHECK_GE(elem_cnt_a, 0);
    CHECK_GE(elem_cnt_b, 0);
    if (elem_cnt_a == 0 || elem_cnt_b == 0) { return; }

    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    CHECK_EQ(out->shape_view().NumAxes(), 2);
    CHECK_EQ(out->data_type(), data_type);
    size_t m = 0, n = 0, k = 0;
    InferMatmulMNK(a->shape_view(), b->shape_view(), out->shape_view(), trans_a, trans_b, &m, &n,
                   &k);
    const double alpha = ctx->Attr<double>("alpha");
    double beta = 0.0;
    if (ctx->has_input("_add_to_output", 0)) {
      const user_op::Tensor* add_to_output = ctx->Tensor4ArgNameAndIndex("_add_to_output", 0);
      CHECK_EQ(add_to_output->data_type(), data_type);
      CHECK_EQ(add_to_output->shape_view(), out->shape_view());
      auto memcpy = NewMemcpyPrimitive(ctx);
      CHECK(memcpy);
      memcpy->Launch(ctx->stream(), out->mut_dptr(), add_to_output->dptr(),
                     add_to_output->shape_view().elem_cnt() * GetSizeOfDataType(data_type));
      beta = 1.0;
    }
    auto matmul = NewMatmulPrimitive(ctx);
    CHECK(matmul);
    matmul->Launch(ctx->stream(), m, n, k, alpha, a->dptr(), b->dptr(), beta, out->mut_dptr());
  }
};

REGISTER_USER_KERNEL("matmul")
    .SetCreateFn<MatmulKernel>()
    .SetIsMatchedHob(MemcpyPrimitiveExists() && MatmulPrimitiveExists())
    .SetInplaceProposalFn([](const user_op::InferContext& ctx,
                             const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {
      if (ctx.has_input("_add_to_output", 0)) {
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "_add_to_output", 0, true));
      }
      return Maybe<void>::Ok();
    });

class BatchMatmulKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  BatchMatmulKernel() = default;
  ~BatchMatmulKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto trans_a = GetBlasTransposeType(ctx, "transpose_a");
    const auto trans_b = GetBlasTransposeType(ctx, "transpose_b");
    const user_op::Tensor* a = ctx->Tensor4ArgNameAndIndex("a", 0);
    const DataType data_type = a->data_type();
    const int64_t num_axes = a->shape_view().NumAxes();
    CHECK_GT(num_axes, 2);
    const user_op::Tensor* b = ctx->Tensor4ArgNameAndIndex("b", 0);
    CHECK_EQ(b->data_type(), data_type);
    CHECK_EQ(b->shape_view().NumAxes(), num_axes);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    CHECK_EQ(out->data_type(), data_type);
    CHECK_EQ(out->shape_view().NumAxes(), num_axes);
    size_t m = 0;
    size_t n = 0;
    size_t k = 0;
    InferMatmulMNK(a->shape_view(), b->shape_view(), out->shape_view(), trans_a, trans_b, &m, &n,
                   &k);
    size_t batch_size = 1;
    for (size_t i = 0; i < num_axes - 2; ++i) {
      const int64_t dim_size = a->shape_view().At(i);
      CHECK_GT(dim_size, 0);
      CHECK_EQ(b->shape_view().At(i), dim_size);
      CHECK_EQ(out->shape_view().At(i), dim_size);
      batch_size *= dim_size;
    }
    const double alpha = ctx->Attr<double>("alpha");
    double beta = 0.0;
    if (ctx->has_input("_add_to_output", 0)) {
      const user_op::Tensor* add_to_output = ctx->Tensor4ArgNameAndIndex("_add_to_output", 0);
      CHECK_EQ(add_to_output->data_type(), data_type);
      CHECK_EQ(add_to_output->shape_view(), out->shape_view());
      auto memcpy = NewMemcpyPrimitive(ctx);
      CHECK(memcpy);
      memcpy->Launch(ctx->stream(), out->mut_dptr(), add_to_output->dptr(),
                     add_to_output->shape_view().elem_cnt() * GetSizeOfDataType(data_type));
      beta = 1.0;
    }
    auto batch_matmul = NewBatchMatmulPrimitive(ctx);
    CHECK(batch_matmul);
    batch_matmul->Launch(ctx->stream(), batch_size, m, n, k, alpha, a->dptr(), b->dptr(), beta,
                         out->mut_dptr());
  }
};

REGISTER_USER_KERNEL("batch_matmul")
    .SetCreateFn<BatchMatmulKernel>()
    .SetIsMatchedHob(MemcpyPrimitiveExists() && BatchMatmulPrimitiveExists())
    .SetInplaceProposalFn([](const user_op::InferContext& ctx,
                             const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {
      if (ctx.has_input("_add_to_output", 0)) {
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "_add_to_output", 0, true));
      }
      return Maybe<void>::Ok();
    });

class BroadcastMatmulKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  BroadcastMatmulKernel() = default;
  ~BroadcastMatmulKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    double alpha = ctx->Attr<double>("alpha");

    const user_op::Tensor* a = ctx->Tensor4ArgNameAndIndex("a", 0);
    const user_op::Tensor* b = ctx->Tensor4ArgNameAndIndex("b", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    double beta = 0.0;
    if (ctx->has_input("_add_to_output", 0)) {
      const user_op::Tensor* add_to_output = ctx->Tensor4ArgNameAndIndex("_add_to_output", 0);
      CHECK_EQ(add_to_output->shape_view(), out->shape_view());
      auto memcpy = NewMemcpyPrimitive(ctx);
      CHECK(memcpy);
      memcpy->Launch(
          ctx->stream(), out->mut_dptr(), add_to_output->dptr(),
          add_to_output->shape_view().elem_cnt() * GetSizeOfDataType(add_to_output->data_type()));
      beta = 1.0;
    }

    const int64_t a_num_axes = a->shape_view().NumAxes();
    const int64_t b_num_axes = b->shape_view().NumAxes();
    const int64_t out_num_axes = out->shape_view().NumAxes();
    auto broadcast_matmul = NewBroadcastMatmulPrimitive(ctx);
    CHECK(broadcast_matmul);
    broadcast_matmul->Launch(ctx->stream(), alpha, a_num_axes, a->shape_view().ptr(), a->dptr(),
                             b_num_axes, b->shape_view().ptr(), b->dptr(), beta, out_num_axes,
                             out->shape_view().ptr(), out->mut_dptr());
  }
};

REGISTER_USER_KERNEL("broadcast_matmul")
    .SetCreateFn<BroadcastMatmulKernel>()
    .SetIsMatchedHob(MemcpyPrimitiveExists() && BroadcastMatmulPrimitiveExists())
    .SetInplaceProposalFn([](const user_op::InferContext& ctx,
                             const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {
      if (ctx.has_input("_add_to_output", 0)) {
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "_add_to_output", 0, true));
      }
      return Maybe<void>::Ok();
    });

template<typename Context>
std::unique_ptr<ep::primitive::Matmul> NewMatmulPrimitiveForBroadcastMatmulGradB(Context* ctx) {
  const DataType data_type = ctx->TensorDesc4ArgNameAndIndex("out", 0)->data_type();
  return NewMatmulPrimitive(ctx->device_type(), data_type, true, false);
}

class BroadcastMatmulGradBKernel final : public user_op::OpKernel,
                                         public user_op::CudaGraphSupport {
 public:
  BroadcastMatmulGradBKernel() = default;
  ~BroadcastMatmulGradBKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    double alpha = ctx->Attr<double>("alpha");
    const user_op::Tensor* a = ctx->Tensor4ArgNameAndIndex("a", 0);
    const user_op::Tensor* b = ctx->Tensor4ArgNameAndIndex("b", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    double beta = 0.0;
    if (ctx->has_input("_add_to_output", 0)) {
      const user_op::Tensor* add_to_output = ctx->Tensor4ArgNameAndIndex("_add_to_output", 0);
      CHECK_EQ(add_to_output->shape_view(), out->shape_view());
      auto memcpy = NewMemcpyPrimitive(ctx);
      CHECK(memcpy);
      memcpy->Launch(
          ctx->stream(), out->mut_dptr(), add_to_output->dptr(),
          add_to_output->shape_view().elem_cnt() * GetSizeOfDataType(add_to_output->data_type()));
      beta = 1.0;
    }

    CHECK_EQ(a->shape_view().NumAxes(), b->shape_view().NumAxes());
    int64_t k = a->shape_view().Count(0, a->shape_view().NumAxes() - 1);
    CHECK_EQ(b->shape_view().Count(0, b->shape_view().NumAxes() - 1), k);
    int64_t m = a->shape_view().At(a->shape_view().NumAxes() - 1);
    int64_t n = b->shape_view().At(b->shape_view().NumAxes() - 1);
    auto matmul = NewMatmulPrimitiveForBroadcastMatmulGradB(ctx);
    CHECK(matmul);
    matmul->Launch(ctx->stream(), m, n, k, alpha, a->dptr(), b->dptr(), beta, out->mut_dptr());
  }
};

auto PrimitiveExistsForBroadcastMatmulGradB() {
  return hob::make_custom("MatmulPrimitiveExists", [](const user_op::KernelRegContext& ctx) {
    return NewMatmulPrimitiveForBroadcastMatmulGradB(&ctx).operator bool();
  });
}

REGISTER_USER_KERNEL("broadcast_matmul_grad_b")
    .SetCreateFn<BroadcastMatmulGradBKernel>()
    .SetIsMatchedHob(MemcpyPrimitiveExists() && PrimitiveExistsForBroadcastMatmulGradB())
    .SetInplaceProposalFn([](const user_op::InferContext& ctx,
                             const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {
      if (ctx.has_input("_add_to_output", 0)) {
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "_add_to_output", 0, true));
      }
      return Maybe<void>::Ok();
    });
}  // namespace

}  // namespace oneflow
