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
#include "oneflow/user/ops/nn_util.h"
#include "oneflow/core/primitive/include/copy_nd.h"
#include "oneflow/core/primitive/include/fill.h"

namespace oneflow {

namespace {

template<typename Context>
std::unique_ptr<primitive::Fill> NewFillPrimitive(Context* ctx) {
  const DataType data_type = ctx->TensorDesc4ArgNameAndIndex("y", 0)->data_type();
  return primitive::NewPrimitive<primitive::FillFactory>(ctx->device_type(), data_type);
}

template<typename Context>
std::unique_ptr<primitive::CopyNd> NewCopyNdPrimitive(Context* ctx) {
  const auto& in_arg_pair = ctx->inputs().front();
  const int64_t ndims =
      ctx->TensorDesc4ArgNameAndIndex(in_arg_pair.first, in_arg_pair.second)->shape().NumAxes();
  return primitive::NewPrimitive<primitive::CopyNdFactory>(ctx->device_type(), ndims);
}

hob::HobContextGetter<user_op::KernelRegContext, bool> FillPrimitiveExists() {
  return user_op::HobCtxGetter<bool>(
      "FillPrimitiveExists",
      [](const user_op::KernelRegContext& ctx) { return NewFillPrimitive(&ctx).operator bool(); });
}

hob::HobContextGetter<user_op::KernelRegContext, bool> CopyNdPrimitiveExists() {
  return user_op::HobCtxGetter<bool>("CopyNdPrimitiveExists",
                                     [](const user_op::KernelRegContext& ctx) {
                                       return NewCopyNdPrimitive(&ctx).operator bool();
                                     });
}

}  // namespace

class SamePaddingKernel final : public user_op::OpKernel {
 public:
  SamePaddingKernel() = default;
  ~SamePaddingKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const int64_t num_axes = x->shape().NumAxes();
    const std::string& padding = ctx->Attr<std::string>("padding");
    const std::string& data_format = ctx->Attr<std::string>("data_format");
    const std::vector<int32_t> kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
    const std::vector<int32_t> strides = ctx->Attr<std::vector<int32_t>>("strides");
    const std::vector<int32_t> dilation_rate = ctx->Attr<std::vector<int32_t>>("dilation_rate");
    std::vector<int64_t> padding_before(num_axes, 0);
    const size_t idx_offset = IdxOffset(data_format);
    const int32_t num_spatial_dims = x->shape().NumAxes() - 2;
    for (int32_t i = 0; i < num_spatial_dims; ++i) {
      int32_t padding_small = 0;
      int32_t padding_large = 0;
      CHECK_JUST(CalcSamePadding(x->shape().At(idx_offset + i), kernel_size.at(i),
                                 dilation_rate.at(i), strides.at(i), &padding_small,
                                 &padding_large));
      if (padding == "same_lower") {
        padding_before[idx_offset + i] = padding_large;
      } else if (padding == "same_upper") {
        padding_before[idx_offset + i] = padding_small;
      } else {
        UNIMPLEMENTED();
      }
      CHECK_EQ(y->shape().At(idx_offset + i),
               x->shape().At(idx_offset + i) + padding_small + padding_large);
    }
    CHECK_EQ(padding_before.size(), num_axes);
    std::unique_ptr<primitive::Fill> fill_primitive = NewFillPrimitive(ctx);
    CHECK(fill_primitive);
    fill_primitive->Launch(ctx->stream_ctx(), y->mut_dptr(), Scalar(0), y->shape().elem_cnt());
    DimVector src_pos_vec(num_axes, 0);
    DimVector dst_pos_vec(padding_before.cbegin(), padding_before.cend());
    std::unique_ptr<primitive::CopyNd> copy_nd_primitive = NewCopyNdPrimitive(ctx);
    CHECK(copy_nd_primitive);
    copy_nd_primitive->Launch(ctx->stream_ctx(), x->data_type(), num_axes, y->mut_dptr(),
                              y->shape().ptr(), dst_pos_vec.data(), x->dptr(), x->shape().ptr(),
                              src_pos_vec.data(), x->shape().ptr());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("same_padding")
    .SetCreateFn<SamePaddingKernel>()
    .SetIsMatchedHob((FillPrimitiveExists() == true) & (CopyNdPrimitiveExists() == true));

class SamePaddingGradKernel final : public user_op::OpKernel {
 public:
  SamePaddingGradKernel() = default;
  ~SamePaddingGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const int64_t num_axes = dy->shape().NumAxes();
    const std::string& padding = ctx->Attr<std::string>("padding");
    const std::string& data_format = ctx->Attr<std::string>("data_format");
    const std::vector<int32_t> kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
    const std::vector<int32_t> strides = ctx->Attr<std::vector<int32_t>>("strides");
    const std::vector<int32_t> dilation_rate = ctx->Attr<std::vector<int32_t>>("dilation_rate");
    std::vector<int64_t> padding_before(num_axes, 0);
    const size_t idx_offset = IdxOffset(data_format);
    const int32_t num_spatial_dims = dy->shape().NumAxes() - 2;
    for (int32_t i = 0; i < num_spatial_dims; ++i) {
      int32_t padding_small = 0;
      int32_t padding_large = 0;
      CHECK_JUST(CalcSamePadding(dx->shape().At(idx_offset + i), kernel_size.at(i),
                                 dilation_rate.at(i), strides.at(i), &padding_small,
                                 &padding_large));
      if (padding == "same_lower") {
        padding_before[idx_offset + i] = padding_large;
      } else if (padding == "same_upper") {
        padding_before[idx_offset + i] = padding_small;
      } else {
        UNIMPLEMENTED();
      }
      CHECK_EQ(dy->shape().At(idx_offset + i),
               dx->shape().At(idx_offset + i) + padding_small + padding_large);
    }
    DimVector dst_pos_vec(num_axes, 0);
    DimVector src_pos_vec(padding_before.cbegin(), padding_before.cend());
    std::unique_ptr<primitive::CopyNd> primitive = NewCopyNdPrimitive(ctx);
    CHECK(primitive);
    primitive->Launch(ctx->stream_ctx(), dy->data_type(), num_axes, dx->mut_dptr(),
                      dx->shape().ptr(), dst_pos_vec.data(), dy->dptr(), dy->shape().ptr(),
                      src_pos_vec.data(), dx->shape().ptr());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("same_padding_grad")
    .SetCreateFn<SamePaddingGradKernel>()
    .SetIsMatchedHob(CopyNdPrimitiveExists() == true);

}  // namespace oneflow
