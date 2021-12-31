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
    // const int64_t axis = ctx->Attr<int64_t>("axis");
    // DimVector dim_vec;
    // for (const auto& in_arg_pair : ctx->inputs()) {
    //   const ShapeView& input_shape_view =
    //       ctx->ShapeView4ArgNameAndIndex(in_arg_pair.first, in_arg_pair.second);
    //   if (dim_vec.size() == 0) {
    //     input_shape_view.ToDimVector(&dim_vec);
    //   } else {
    //     CHECK_EQ(input_shape_view.NumAxes(), dim_vec.size());
    //     FOR_RANGE(int64_t, i, 0, input_shape_view.NumAxes()) {
    //       if (i == axis) {
    //         dim_vec.at(i) += input_shape_view.At(i);
    //       } else {
    //         CHECK_EQ(input_shape_view.At(i), dim_vec.at(i));
    //       }
    //     }
    //   }

    const ShapeView& first_input_shape_view = ctx->ShapeView4ArgNameAndIndex("in", 0);
    const int64_t axis = ctx->Attr<int64_t>("axis");
    const int64_t in_num_axes = first_input_shape_view.NumAxes(); 
    DimVector out_dim_vec(in_num_axes+1); 
    for(int i = 0; i < in_num_axes+1; i++){
        if(i == axis){
            continue; 
        }else{
            out_dim_vec.at(i) = first_input_shape_view.At(i);
        }
    }
  for (const auto& in_arg_pair : ctx->inputs()) {
    const ShapeView& input_shape_view = ctx->ShapeView4ArgNameAndIndex(in_arg_pair.first, in_arg_pair.second);
    CHECK_EQ(input_shape_view.NumAxes(), first_input_shape_view.NumAxes());
    FOR_RANGE(int64_t, i, 0, in_num_axes+1) {
      if (i == axis) {
        out_dim_vec.at(axis) += 1; 
      } else if(i < axis) {
        CHECK_EQ(input_shape_view.At(i), out_dim_vec.at(i));
      }else{
        CHECK_EQ(input_shape_view.At(i-1), out_dim_vec.at(i));
      }
    }
  }

    ctx->MutShapeView4ArgNameAndIndex("out", 0)->set_shape(Shape(out_dim_vec));
  }

  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    if (out_tensor->shape().elem_cnt() == 0) { return; }
    const int64_t axis = ctx->Attr<int64_t>("axis");
    printf("Axis is: %ld \n", axis); 
    const int64_t out_cols = out_tensor->shape().Count(axis);
    // const int64_t rows = out_tensor->shape().elem_cnt() / out_cols;
    const int64_t rows = out_tensor->shape().elem_cnt() / out_tensor->shape().Count(axis);
    CHECK_GT(rows, 0);
    printf("Out cols is: %ld \n", out_cols); 
    auto primitive = NewCopyNdPrimitive(ctx);
    CHECK(primitive);
    int64_t out_col_offset = 0;
    for (const auto& in_arg_pair : ctx->inputs()) {
      const user_op::Tensor* in_tensor =
          ctx->Tensor4ArgNameAndIndex(in_arg_pair.first, in_arg_pair.second);
      if (in_tensor->shape().elem_cnt() == 0) { continue; }
      const int64_t in_cols = in_tensor->shape().Count(axis);
      printf("In cols is: %ld \n", in_cols); 
      CHECK_EQ(in_tensor->shape().elem_cnt(), rows * in_cols);
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

REGISTER_USER_KERNEL("stack").SetCreateFn<StackKernel>().SetIsMatchedHob(CopyNdPrimitiveExists()
                                                                           == true);

// template<typename Context>
// std::unique_ptr<ep::primitive::CopyNd> NewCopyNdPrimitive(Context* ctx) {
//   return ep::primitive::NewPrimitive<ep::primitive::CopyNdFactory>(ctx->device_type(), 2);
// }

class StackBackwardKernel final : public user_op::OpKernel {
 public:
  StackBackwardKernel() = default;
  ~StackBackwardKernel() override = default;

 private:
  void InferShape(user_op::KernelInferContext* ctx) const override {
    const auto axis = ctx->Attr<int64_t>("axis");
    const ShapeView& in_shape_view = ctx->ShapeView4ArgNameAndIndex("in", 0);
    int64_t total_dim_size = 0;
    const int64_t like_num_axes = ctx->ShapeView4ArgNameAndIndex("like", 0).NumAxes();
    const int64_t in_num_axes = in_shape_view.NumAxes();
    CHECK_LE(like_num_axes, in_num_axes);
    CHECK_LT(axis, like_num_axes);
    FOR_RANGE(int32_t, i, 0, ctx->outputs().size()) {
      const ShapeView& like_shape_view = ctx->ShapeView4ArgNameAndIndex("like", i);
      CHECK_EQ(like_shape_view.NumAxes(), like_num_axes);
      // FOR_RANGE(int64_t, j, 0, like_num_axes) {
      //   if (j == axis) {
      //     total_dim_size += like_shape_view.At(j);
      //   } else {
      //     CHECK_EQ(like_shape_view.At(j), in_shape_view.At(j));
      //   }
      // }

      FOR_RANGE(int64_t, j, 0, like_num_axes+1) {
      if (j == axis) {
          total_dim_size += like_shape_view.Count(j);
      } else if (j < axis){
        CHECK_EQ(in_shape_view.At(j), like_shape_view.At(j));
      } else {
        CHECK_EQ(in_shape_view.At(j), like_shape_view.At(j-1));
      }
    }

      if (ctx->TensorDesc4ArgNameAndIndex("out", i)->is_dynamic()) {
        auto* mut_shape_view = ctx->MutShapeView4ArgNameAndIndex("out", i);
        CHECK_NOTNULL(mut_shape_view);
        DimVector out_i_dim_vec;
        like_shape_view.ToDimVector(&out_i_dim_vec);
        // no need ？
        // FOR_RANGE(int64_t, j, like_num_axes, in_num_axes) {
        //   out_i_dim_vec.emplace_back(in_shape_view.At(j));
        // }
        mut_shape_view->set_shape(Shape(out_i_dim_vec));
      }
    }
    CHECK_EQ(total_dim_size, in_shape_view.Count(axis));
  }

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in_tensor = ctx->Tensor4ArgNameAndIndex("in", 0);
    const auto axis = ctx->Attr<int64_t>("axis");
    const int64_t in_cols = in_tensor->shape().Count(axis);
    const int64_t rows = in_tensor->shape().elem_cnt() / in_cols;
    CHECK_GT(rows, 0);

    auto primitive = NewCopyNdPrimitive(ctx);
    CHECK(primitive);
    int64_t in_col_offset = 0;
    for (const auto& out_arg_pair : ctx->outputs()) {
      user_op::Tensor* out_tensor =
          ctx->Tensor4ArgNameAndIndex(out_arg_pair.first, out_arg_pair.second);
      const int64_t out_cols = out_tensor->shape().Count(axis);
      CHECK_EQ(out_tensor->shape().elem_cnt(), rows * out_cols);
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
    CHECK_EQ(in_col_offset, in_cols);
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

// auto CopyNdPrimitiveExists() {
//   return hob::make_custom("CopyNdPrimitiveExists", [](const user_op::KernelRegContext& ctx) {
//     return NewCopyNdPrimitive(&ctx).operator bool();
//   });
// }

REGISTER_USER_KERNEL("stack_backward")
    .SetCreateFn<StackBackwardKernel>()
    .SetIsMatchedHob(CopyNdPrimitiveExists() == true);
}  // namespace oneflow
