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
#include "oneflow/core/ep/include/primitive/copy_nd.h"
#include "oneflow/core/ep/include/primitive/memset.h"

namespace oneflow {

namespace user_op {

namespace {

template<typename Context>
std::unique_ptr<ep::primitive::CopyNd> NewCopyNdPrimitive(Context* ctx) {
  return ep::primitive::NewPrimitive<ep::primitive::CopyNdFactory>(ctx->device_type(), 3);
}

template<typename Context>
std::unique_ptr<ep::primitive::Memset> NewMemsetPrimitive(Context* ctx) {
  return ep::primitive::NewPrimitive<ep::primitive::MemsetFactory>(ctx->device_type());
}

auto CopyNdPrimitiveExists() {
  return hob::make_custom("CopyNdPrimitiveExists", [](const user_op::KernelRegContext& ctx) {
    return NewCopyNdPrimitive(&ctx).operator bool();
  });
}

auto MemsetPrimitiveExists() {
  return hob::make_custom("MemsetPrimitiveExists", [](const KernelRegContext& ctx) {
    return NewMemsetPrimitive(&ctx).operator bool();
  });
}

}  // namespace

class NarrowKernel final : public user_op::OpKernel {
 public:
  NarrowKernel() = default;
  ~NarrowKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    if (in->shape_view().elem_cnt() == 0) { return; }
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int64_t& dim = ctx->Attr<int64_t>("dim");
    const int64_t& start = ctx->Attr<int64_t>("start");
    int64_t length = out->shape_view().At(dim);
    const ShapeView in_shape = in->shape_view();
    auto copy_nd_primitive = NewCopyNdPrimitive(ctx);
    CHECK(copy_nd_primitive);

    const int64_t outer_dim = in_shape.Count(0, dim);
    const int64_t inner_dim = in_shape.Count(dim + 1);
    const int64_t narrow_dim = in_shape.At(dim);

    DimVector dst_shape = {outer_dim, length, inner_dim};
    DimVector dst_pos_vec = {0, 0, 0};

    DimVector src_shape = {outer_dim, narrow_dim, inner_dim};
    DimVector src_pos_vec = {0, start, 0};
    DimVector extent_vec = {outer_dim, length, inner_dim};
    copy_nd_primitive->Launch(ctx->stream(), out->data_type(), 3, out->mut_dptr(), dst_shape.data(),
                              dst_pos_vec.data(), in->dptr(), src_shape.data(), src_pos_vec.data(),
                              extent_vec.data());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

class NarrowGradKernel final : public user_op::OpKernel {
 public:
  NarrowGradKernel() = default;
  ~NarrowGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const int64_t& dim = ctx->Attr<int64_t>("dim");
    const int64_t& start = ctx->Attr<int64_t>("start");
    int64_t length = dy->shape_view().At(dim);

    size_t dx_byte_size = dx->shape_view().elem_cnt() * GetSizeOfDataType(dx->data_type());
    void* dst = dx->mut_dptr();
    std::unique_ptr<ep::primitive::Memset> memset_primitive =
        ep::primitive::NewPrimitive<ep::primitive::MemsetFactory>(ctx->device_type());
    CHECK(memset_primitive);
    memset_primitive->Launch(ctx->stream(), dst, 0, dx_byte_size);

    auto copy_nd_primitive = NewCopyNdPrimitive(ctx);
    CHECK(copy_nd_primitive);
    const ShapeView dx_shape = dx->shape_view();

    const int64_t outer_dim = dx_shape.Count(0, dim);
    const int64_t inner_dim = dx_shape.Count(dim + 1);
    const int64_t narrow_dim = dx_shape.At(dim);

    DimVector dst_shape = {outer_dim, narrow_dim, inner_dim};
    DimVector dst_pos_vec = {0, start, 0};

    DimVector src_shape = {outer_dim, length, inner_dim};
    DimVector src_pos_vec = {0, 0, 0};
    DimVector extent_vec = {outer_dim, length, inner_dim};

    copy_nd_primitive->Launch(ctx->stream(), dx->data_type(), 3, dst, dst_shape.data(),
                              dst_pos_vec.data(), dy->dptr(), src_shape.data(), src_pos_vec.data(),
                              extent_vec.data());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("narrow").SetCreateFn<NarrowKernel>().SetIsMatchedHob(CopyNdPrimitiveExists()
                                                                           == true);
REGISTER_USER_KERNEL("narrow_grad")
    .SetCreateFn<NarrowGradKernel>()
    .SetIsMatchedHob(MemsetPrimitiveExists() && CopyNdPrimitiveExists());

}  // namespace user_op

}  // namespace oneflow
