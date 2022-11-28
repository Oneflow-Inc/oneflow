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

namespace oneflow {

namespace {

template<typename Context>
std::unique_ptr<ep::primitive::CopyNd> NewCopyNdPrimitive(Context* ctx) {
  return ep::primitive::NewPrimitive<ep::primitive::CopyNdFactory>(ctx->device_type(), 2);
}

class StackKernel final : public user_op::OpKernel {
 public:
  StackKernel() = default;
  ~StackKernel() = default;

 private:
  void InferShape(user_op::KernelInferContext* ctx) const override {
    const ShapeView& first_input_shape_view = ctx->ShapeView4ArgNameAndIndex("in", 0);
    const int64_t axis = ctx->Attr<int64_t>("axis");
    const int64_t in_num_axes = first_input_shape_view.NumAxes();
    DimVector out_dim_vec(in_num_axes + 1);
    for (int i = 0; i < in_num_axes + 1; i++) {
      if (i == axis) {
        continue;
      } else {
        out_dim_vec.at(i) = first_input_shape_view.At(i);
      }
    }
    for (const auto& in_arg_pair : ctx->inputs()) {
      const ShapeView& input_shape_view =
          ctx->ShapeView4ArgNameAndIndex(in_arg_pair.first, in_arg_pair.second);
      CHECK_EQ(input_shape_view.NumAxes(), first_input_shape_view.NumAxes());
      FOR_RANGE(int64_t, i, 0, in_num_axes + 1) {
        if (i == axis) {
          out_dim_vec.at(axis) += 1;
        } else if (i < axis) {
          CHECK_EQ(input_shape_view.At(i), out_dim_vec.at(i))
              << " Stack expects each tensor to be equal size"
                 ", but got "
              << first_input_shape_view.ToString() << " at first input and "
              << input_shape_view.ToString();
        } else {
          CHECK_EQ(input_shape_view.At(i - 1), out_dim_vec.at(i))
              << " Stack expects each tensor to be equal size"
                 ", but got "
              << first_input_shape_view.ToString() << " at first input and "
              << input_shape_view.ToString();
        }
      }
    }

    ctx->MutShapeView4ArgNameAndIndex("out", 0).set_shape(Shape(out_dim_vec));
  }

  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    if (out_tensor->shape_view().elem_cnt() == 0) { return; }
    const int64_t axis = ctx->Attr<int64_t>("axis");
    const int64_t out_cols = out_tensor->shape_view().Count(axis);
    const int64_t rows = out_tensor->shape_view().Count(0, axis);
    CHECK_GT(rows, 0) << "The multiplicative from axis 0 to axis " << axis - 1
                      << " should be greater than 0. ";
    auto primitive = NewCopyNdPrimitive(ctx);
    CHECK(primitive) << "Error in Stack kernel NewCopyNdPrimitive. ";
    int64_t out_col_offset = 0;
    for (const auto& in_arg_pair : ctx->inputs()) {
      const user_op::Tensor* in_tensor =
          ctx->Tensor4ArgNameAndIndex(in_arg_pair.first, in_arg_pair.second);
      if (in_tensor->shape_view().elem_cnt() == 0) { continue; }
      const int64_t in_cols = in_tensor->shape_view().Count(axis);
      CHECK_EQ(in_tensor->shape_view().elem_cnt(), rows * in_cols)
          << "The element count of input tensor is not equal to `rows * in_cols`. ";
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
    CHECK_EQ(out_col_offset, out_cols) << "The out column offset is not equal to out columns. ";
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

REGISTER_USER_KERNEL("stack").SetCreateFn<StackKernel>().SetIsMatchedHob(CopyNdPrimitiveExists()
                                                                         == true);

class StackGradKernel final : public user_op::OpKernel {
 public:
  StackGradKernel() = default;
  ~StackGradKernel() override = default;

 private:
  void InferShape(user_op::KernelInferContext* ctx) const override {
    const auto axis = ctx->Attr<int64_t>("axis");
    const ShapeView& in_shape_view = ctx->ShapeView4ArgNameAndIndex("in", 0);
    int64_t total_dim_size = 0;
    const int64_t like_num_axes = ctx->ShapeView4ArgNameAndIndex("like", 0).NumAxes();
    const int64_t in_num_axes = in_shape_view.NumAxes();
    CHECK_LE(like_num_axes, in_num_axes)
        << "The num axes of `like` tensor should be less equal to num axes of `in` tensor. ";
    CHECK_LE(axis, like_num_axes)
        << "The axis should be less than or equal to num axes of `like` tensor. ";
    FOR_RANGE(size_t, i, 0, ctx->outputs().size()) {
      const ShapeView& like_shape_view = ctx->ShapeView4ArgNameAndIndex("like", i);
      CHECK_EQ(like_shape_view.NumAxes(), like_num_axes)
          << "The num axes of `like` tensor at index " << i
          << " should be equal to first `like` tensor. ";
      FOR_RANGE(int64_t, j, 0, like_num_axes + 1) {
        if (j == axis) {
          total_dim_size += like_shape_view.Count(j);
        } else if (j < axis) {
          CHECK_EQ(in_shape_view.At(j), like_shape_view.At(j))
              << " Stack Grad expects the shape of input tensor is equal to like tensor's. "
                 ", but got "
              << in_shape_view.ToString() << " at input and " << like_shape_view.ToString()
              << "at like ";
        } else {
          CHECK_EQ(in_shape_view.At(j), like_shape_view.At(j - 1))
              << " Stack Grad expects the shape of input tensor is equal to like tensor's. "
                 ", but got "
              << in_shape_view.ToString() << " at input and " << like_shape_view.ToString()
              << "at like ";
        }
      }

      if (ctx->TensorDesc4ArgNameAndIndex("out", i)->is_dynamic()) {
        auto mut_shape_view = ctx->MutShapeView4ArgNameAndIndex("out", i);
        DimVector out_i_dim_vec;
        like_shape_view.ToDimVector(&out_i_dim_vec);
        mut_shape_view.set_shape(Shape(out_i_dim_vec));
      }
    }
    CHECK_EQ(total_dim_size, in_shape_view.Count(axis))
        << "The sum of dim size of each `like` tensor should be equal to `in` tensor count from "
           "axis "
        << axis;
  }

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in_tensor = ctx->Tensor4ArgNameAndIndex("in", 0);
    const int64_t axis = ctx->Attr<int64_t>("axis");
    const int64_t in_cols = in_tensor->shape_view().Count(axis);
    const int64_t rows = in_tensor->shape_view().Count(0, axis);
    CHECK_GT(rows, 0) << "The multiplicative from axis 0 to axis " << axis - 1
                      << " should be greater than 0. ";
    auto primitive = NewCopyNdPrimitive(ctx);
    CHECK(primitive) << "Error in Stack Grad kernel NewCopyNdPrimitive. ";
    int64_t in_col_offset = 0;
    for (const auto& out_arg_pair : ctx->outputs()) {
      user_op::Tensor* out_tensor =
          ctx->Tensor4ArgNameAndIndex(out_arg_pair.first, out_arg_pair.second);
      const int64_t out_cols = out_tensor->shape_view().Count(axis);
      CHECK_EQ(out_tensor->shape_view().elem_cnt(), rows * out_cols)
          << "The element count of output tensor is not equal to `rows * out_cols`. ";
      if (out_cols > 0) {
        DimVector dst_shape = {rows, out_cols};
        DimVector dst_pos_vec = {0, 0};
        DimVector src_shape = {rows, in_cols};
        DimVector src_pos_vec = {0, in_col_offset};
        DimVector extent_vec = {rows, out_cols};
        primitive->Launch(ctx->stream(), out_tensor->data_type(), 2, out_tensor->mut_dptr(),
                          dst_shape.data(), dst_pos_vec.data(), in_tensor->dptr(), src_shape.data(),
                          src_pos_vec.data(), extent_vec.data());
      }
      in_col_offset += out_cols;
    }
    CHECK_EQ(in_col_offset, in_cols) << "The in column offset is not equal to in columns.";
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("stack_grad")
    .SetCreateFn<StackGradKernel>()
    .SetIsMatchedHob(CopyNdPrimitiveExists() == true);
}  // namespace oneflow
