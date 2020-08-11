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
class ConcatKernel final : public user_op::OpKernel {
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
    ctx->MutShapeView4ArgNameAndIndex("out", 0)->set_shape(Shape(dim_vec));
  }

  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int64_t axis = ctx->Attr<int64_t>("axis");
    const int64_t out_cols = out_tensor->shape().Count(axis);
    const int64_t rows = out_tensor->shape().elem_cnt() / out_cols;
    CHECK_GT(rows, 0);
    int64_t out_col_offset = 0;
    for (const auto& in_arg_pair : ctx->inputs()) {
      const user_op::Tensor* in_tensor =
          ctx->Tensor4ArgNameAndIndex(in_arg_pair.first, in_arg_pair.second);
      const int64_t in_cols = in_tensor->shape().Count(axis);
      CHECK_EQ(in_tensor->shape().elem_cnt(), rows * in_cols);
      if (in_cols > 0) {
        NewKernelUtil<device_type>::CopyColsRegion(
            ctx->device_ctx(), rows, in_cols, in_tensor->dptr<T>(), 0, in_cols,
            out_tensor->mut_dptr<T>(), out_col_offset, out_cols);
      }
      out_col_offset += in_cols;
    }
    CHECK_EQ(out_col_offset, out_cols);
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

#define REGISTER_CONCAT_KERNEL(device, dtype)                                                \
  REGISTER_USER_KERNEL("concat").SetCreateFn<ConcatKernel<device, dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceTag() == device)                                                    \
      & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

#define REGISTER_CONCAT_KERNEL_WITH_DEVICE(device) \
  REGISTER_CONCAT_KERNEL(device, float)            \
  REGISTER_CONCAT_KERNEL(device, double)           \
  REGISTER_CONCAT_KERNEL(device, int8_t)           \
  REGISTER_CONCAT_KERNEL(device, int32_t)          \
  REGISTER_CONCAT_KERNEL(device, int64_t)

REGISTER_CONCAT_KERNEL_WITH_DEVICE(DeviceType::kCPU)
#ifdef WITH_CUDA
REGISTER_CONCAT_KERNEL_WITH_DEVICE(DeviceType::kGPU)
REGISTER_CONCAT_KERNEL(DeviceType::kGPU, float16)
#endif

}  // namespace oneflow
