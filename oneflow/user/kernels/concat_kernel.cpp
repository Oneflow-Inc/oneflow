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
#include "oneflow/core/kernel/cuda_graph_support.h"

namespace oneflow {

namespace {

template<typename Context>
std::unique_ptr<ep::primitive::CopyNd> NewCopyNdPrimitive(Context* ctx) {
  return ep::primitive::NewPrimitive<ep::primitive::CopyNdFactory>(ctx->device_type(), 2);
}

class ConcatKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  ConcatKernel() = default;
  ~ConcatKernel() = default;

 private:
  void InferShape(user_op::KernelInferContext* ctx) const override {
    const int64_t axis = ctx->Attr<int64_t>("axis");
    DimVector dim_vec;
    for (const auto& in_arg_pair : ctx->inputs()) {
      const ShapeView& input_shape_view =
          ctx->ShapeView4ArgNameAndIndex(in_arg_pair.first, in_arg_pair.second);
      if (dim_vec.size() == 0) {
        input_shape_view.ToDimVector(&dim_vec);
      } else {
        CHECK_EQ(input_shape_view.NumAxes(), dim_vec.size());
        FOR_RANGE(int64_t, i, 0, input_shape_view.NumAxes()) {
          if (i == axis) {
            dim_vec.at(i) += input_shape_view.At(i);
          } else {
            CHECK_EQ(input_shape_view.At(i), dim_vec.at(i));
          }
        }
      }
    }
    ctx->MutShapeView4ArgNameAndIndex("out", 0).set_shape(Shape(dim_vec));
  }

  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    if (out_tensor->shape_view().elem_cnt() == 0) { return; }
    const int64_t axis = ctx->Attr<int64_t>("axis");
    const int64_t out_cols = out_tensor->shape_view().Count(axis);
    const int64_t rows = out_tensor->shape_view().elem_cnt() / out_cols;
    CHECK_GT(rows, 0);

    auto primitive = NewCopyNdPrimitive(ctx);
    CHECK(primitive);
    int64_t out_col_offset = 0;
    for (const auto& in_arg_pair : ctx->inputs()) {
      const user_op::Tensor* in_tensor =
          ctx->Tensor4ArgNameAndIndex(in_arg_pair.first, in_arg_pair.second);
      if (in_tensor->shape_view().elem_cnt() == 0) { continue; }
      const int64_t in_cols = in_tensor->shape_view().Count(axis);
      CHECK_EQ(in_tensor->shape_view().elem_cnt(), rows * in_cols);
      if (in_cols > 0) {
        DimVector dst_shape = {rows, out_cols};
        DimVector dst_pos_vec = {0, out_col_offset};
        DimVector src_shape = {rows, in_cols};
        DimVector src_pos_vec = {0, 0};
        DimVector extent_vec = {rows, in_cols};
        primitive->Launch(ctx->stream(), out_tensor->data_type(), 2, out_tensor->mut_dptr(),
                          dst_shape.data(), dst_pos_vec.data(), in_tensor->dptr(), src_shape.data(),
                          src_pos_vec.data(), extent_vec.data());
      }
      out_col_offset += in_cols;
    }
    CHECK_EQ(out_col_offset, out_cols);
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

auto CopyNdPrimitiveExists() {
  return hob::make_custom("CopyNdPrimitiveExists",
                          [](const user_op::KernelRegContext& ctx) -> bool {
                            return NewCopyNdPrimitive(&ctx).operator bool();
                          });
}

}  // namespace

REGISTER_USER_KERNEL("concat").SetCreateFn<ConcatKernel>().SetIsMatchedHob(CopyNdPrimitiveExists()
                                                                           == true);

}  // namespace oneflow
