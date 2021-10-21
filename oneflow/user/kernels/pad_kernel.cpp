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
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/primitive/include/copy_nd.h"
#include "oneflow/core/primitive/include/fill.h"
#include "oneflow/core/primitive/include/memset.h"

namespace oneflow {

namespace user_op {

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

template<typename Context>
std::unique_ptr<primitive::Memset> NewMemsetPrimitive(Context* ctx) {
  return primitive::NewPrimitive<primitive::MemsetFactory>(ctx->device_type());
}

hob::HobContextGetter<KernelRegContext, bool> FillPrimitiveExists() {
  return HobCtxGetter<bool>("FillPrimitiveExists", [](const KernelRegContext& ctx) {
    return NewFillPrimitive(&ctx).operator bool();
  });
}

hob::HobContextGetter<KernelRegContext, bool> CopyNdPrimitiveExists() {
  return HobCtxGetter<bool>("CopyNdPrimitiveExists", [](const KernelRegContext& ctx) {
    return NewCopyNdPrimitive(&ctx).operator bool();
  });
}

hob::HobContextGetter<KernelRegContext, bool> MemsetPrimitiveExists() {
  return HobCtxGetter<bool>("MemsetPrimitiveExists", [](const KernelRegContext& ctx) {
    return NewMemsetPrimitive(&ctx).operator bool();
  });
}

}  // namespace

class PadKernel final : public OpKernel, public CudaGraphSupport {
 public:
  PadKernel() = default;
  ~PadKernel() = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    if (y->shape().NumAxes() > 0 && y->shape().elem_cnt() == 0) {
      // if output is 0-shape tensor, than do nothing and return
      return;
    }

    Scalar value;
    if (IsIntegralDataType(x->data_type())) {
      value = Scalar(ctx->Attr<int64_t>("integral_constant_value"));
    } else {
      value = Scalar(ctx->Attr<double>("floating_constant_value"));
    }
    std::unique_ptr<primitive::Fill> fill_primitive = NewFillPrimitive(ctx);
    CHECK(fill_primitive);
    fill_primitive->Launch(ctx->stream_ctx(), y->mut_dptr(), value, y->shape().elem_cnt());

    const auto& padding_before = ctx->Attr<std::vector<int64_t>>("padding_before");
    const auto& padding_after = ctx->Attr<std::vector<int64_t>>("padding_after");
    const int64_t ndims = x->shape().NumAxes();
    CHECK_EQ(padding_before.size(), ndims);

    DimVector src_pos_vec(ndims, 0);
    DimVector dst_pos_vec(padding_before.cbegin(), padding_before.cend());
    DimVector pad_before_vec(padding_before.cbegin(), padding_before.cend());
    DimVector pad_after_vec(padding_after.cbegin(), padding_after.cend());

    for (int i = 0; i < ndims; ++i) {
      if (dst_pos_vec[i] < 0) {
        // When padding[i] < 0 , dst_pos_vec[i] will < 0 too , src_pos_vec[i] should adjust coords
        // relative and dst_pos_vec[i] will == 0
        src_pos_vec[i] -= dst_pos_vec[i];
        dst_pos_vec[i] = 0;
      }
    }

    DimVector extent_vec(ndims, 0);
    for (int i = 0; i < extent_vec.size(); ++i) {
      if (y->shape().At(i) < x->shape().At(i)) {
        extent_vec[i] = y->shape().At(i);
      } else {
        extent_vec[i] = x->shape().At(i);
        if (pad_before_vec[i] < 0) { extent_vec[i] = extent_vec[i] + pad_before_vec[i]; }
        if (pad_after_vec[i] < 0) { extent_vec[i] = extent_vec[i] + pad_after_vec[i]; }
      }
    }
    std::unique_ptr<primitive::CopyNd> copy_nd_primitive = NewCopyNdPrimitive(ctx);
    CHECK(copy_nd_primitive);
    copy_nd_primitive->Launch(ctx->stream_ctx(), x->data_type(), x->shape().NumAxes(),
                              y->mut_dptr(), y->shape().ptr(), dst_pos_vec.data(), x->dptr(),
                              x->shape().ptr(), src_pos_vec.data(), extent_vec.data());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("pad").SetCreateFn<PadKernel>().SetIsMatchedHob((FillPrimitiveExists() == true)
                                                                     & (CopyNdPrimitiveExists()
                                                                        == true));

class PadGradKernel final : public OpKernel, public CudaGraphSupport {
 public:
  PadGradKernel() = default;
  ~PadGradKernel() = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    size_t out_bytes_size = dx->shape().elem_cnt() * GetSizeOfDataType(dx->data_type());
    void* dst = dx->mut_dptr();

    std::unique_ptr<primitive::Memset> memset_primitive =
        primitive::NewPrimitive<primitive::MemsetFactory>(ctx->device_type());
    CHECK(memset_primitive);
    memset_primitive->Launch(ctx->stream_ctx(), dst, 0, out_bytes_size);

    if ((dy->shape().NumAxes() > 0 && dy->shape().elem_cnt() == 0)
        || (dx->shape().NumAxes() > 0 && dx->shape().elem_cnt() == 0)) {
      // if input/output is 0-shape tensor, than do nothing and return
      return;
    }

    const auto& padding_before = ctx->Attr<std::vector<int64_t>>("padding_before");
    const auto& padding_after = ctx->Attr<std::vector<int64_t>>("padding_after");
    const int64_t ndims = dy->shape().NumAxes();

    DimVector dst_pos_vec(ndims, 0);
    DimVector src_pos_vec(padding_before.cbegin(), padding_before.cend());
    DimVector pad_before_vec(padding_before.cbegin(), padding_before.cend());
    DimVector pad_after_vec(padding_after.cbegin(), padding_after.cend());

    for (int i = 0; i < ndims; ++i) {
      if (src_pos_vec[i] < 0) {
        dst_pos_vec[i] -= src_pos_vec[i];
        src_pos_vec[i] = 0;
      }
    }

    DimVector extent_vec(ndims, 0);
    for (int i = 0; i < extent_vec.size(); ++i) {
      if (dy->shape().At(i) < dx->shape().At(i)) {
        extent_vec[i] = dy->shape().At(i);
      } else {
        extent_vec[i] = dx->shape().At(i);
        if (pad_before_vec[i] < 0) { extent_vec[i] = extent_vec[i] + pad_before_vec[i]; }
        if (pad_after_vec[i] < 0) { extent_vec[i] = extent_vec[i] + pad_after_vec[i]; }
      }
    }
    std::unique_ptr<primitive::CopyNd> copy_nd_primitive =
        primitive::NewPrimitive<primitive::CopyNdFactory>(ctx->device_type(), ndims);
    CHECK(copy_nd_primitive);
    copy_nd_primitive->Launch(ctx->stream_ctx(), dy->data_type(), ndims, dst, dx->shape().ptr(),
                              dst_pos_vec.data(), dy->dptr(), dy->shape().ptr(), src_pos_vec.data(),
                              extent_vec.data());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("pad_grad")
    .SetCreateFn<PadGradKernel>()
    .SetIsMatchedHob((MemsetPrimitiveExists() == true) & (CopyNdPrimitiveExists() == true));

}  // namespace user_op

}  // namespace oneflow
