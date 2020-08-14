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

namespace oneflow {

namespace {

template<DeviceType device_type, typename T>
class SplitLikeKernel final : public user_op::OpKernel {
 public:
  SplitLikeKernel() = default;
  ~SplitLikeKernel() = default;

 private:
  void InferShape(user_op::KernelInferContext* ctx) const override {
    const int64_t axis = ctx->Attr<int64_t>("axis");
    const ShapeView& in_shape_view = ctx->ShapeView4ArgNameAndIndex("in", 0);
    int64_t total_dim_size = 0;
    FOR_RANGE(int32_t, i, 0, ctx->outputs().size()) {
      const ShapeView& like_shape_view = ctx->ShapeView4ArgNameAndIndex("like", i);
      CHECK_EQ(like_shape_view.NumAxes(), in_shape_view.NumAxes());
      FOR_RANGE(int64_t, j, 0, like_shape_view.NumAxes()) {
        if (j == axis) {
          total_dim_size += like_shape_view.At(j);
        } else {
          CHECK_EQ(like_shape_view.At(j), in_shape_view.At(j));
        }
      }
      if (ctx->TensorDesc4ArgNameAndIndex("out", i)->is_dynamic()) {
        auto* mut_shape_view = ctx->MutShapeView4ArgNameAndIndex("out", i);
        CHECK_NOTNULL(mut_shape_view);
        mut_shape_view->set_shape(like_shape_view);
      }
    }
    CHECK_EQ(total_dim_size, in_shape_view.At(axis));
  }

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in_tensor = ctx->Tensor4ArgNameAndIndex("in", 0);
    const int64_t axis = ctx->Attr<int64_t>("axis");
    const int64_t in_cols = in_tensor->shape().Count(axis);
    const int64_t rows = in_tensor->shape().elem_cnt() / in_cols;
    CHECK_GT(rows, 0);
    int64_t in_col_offset = 0;
    for (const auto& out_arg_pair : ctx->outputs()) {
      user_op::Tensor* out_tensor =
          ctx->Tensor4ArgNameAndIndex(out_arg_pair.first, out_arg_pair.second);
      const int64_t out_cols = out_tensor->shape().Count(axis);
      CHECK_EQ(out_tensor->shape().elem_cnt(), rows * out_cols);
      if (out_cols > 0) {
        NewKernelUtil<device_type>::CopyColsRegion(ctx->device_ctx(), rows, out_cols,
                                                   in_tensor->dptr<T>(), in_col_offset, in_cols,
                                                   out_tensor->mut_dptr<T>(), 0, out_cols);
      }
      in_col_offset += out_cols;
    }
    CHECK_EQ(in_col_offset, in_cols);
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

#define REGISTER_SPLIT_LIKE_KERNEL(device, dtype)          \
  REGISTER_USER_KERNEL("split_like")                       \
      .SetCreateFn<SplitLikeKernel<device, dtype>>()       \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device) \
                       & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

#define REGISTER_SPLIT_LIKE_KERNEL_WITH_DEVICE(device) \
  REGISTER_SPLIT_LIKE_KERNEL(device, float)            \
  REGISTER_SPLIT_LIKE_KERNEL(device, double)           \
  REGISTER_SPLIT_LIKE_KERNEL(device, int8_t)           \
  REGISTER_SPLIT_LIKE_KERNEL(device, int32_t)          \
  REGISTER_SPLIT_LIKE_KERNEL(device, int64_t)

REGISTER_SPLIT_LIKE_KERNEL_WITH_DEVICE(DeviceType::kCPU)
#ifdef WITH_CUDA
REGISTER_SPLIT_LIKE_KERNEL_WITH_DEVICE(DeviceType::kGPU)
REGISTER_SPLIT_LIKE_KERNEL(DeviceType::kGPU, float16)
#endif

}  // namespace oneflow
