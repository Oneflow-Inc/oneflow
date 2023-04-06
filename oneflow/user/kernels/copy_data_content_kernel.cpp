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
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/ep/include/primitive/memcpy.h"
#include "oneflow/core/ep/include/primitive/fill.h"

namespace oneflow {

namespace {

class CopyDataContentKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  CopyDataContentKernel() = default;
  ~CopyDataContentKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int64_t elem_cnt = in->shape_view().elem_cnt();
    // For 0-size tensor, we don't need to copy data, but we must
    // fill output tensor with Scalar(0) because during the backward propogation, this kernel will
    // also be used.
    if (elem_cnt == 0) {
      const int64_t out_elem_cnt = out->shape_view().elem_cnt();
      CHECK_GE(out_elem_cnt, 0);
      if (out_elem_cnt == 0) { return; }
      std::unique_ptr<ep::primitive::Fill> fill =
          ep::primitive::NewPrimitive<ep::primitive::FillFactory>(ctx->device_type(),
                                                                  out->data_type());
      CHECK(fill);
      fill->Launch(ctx->stream(), out->mut_dptr(), Scalar(0), out_elem_cnt);
      return;
    }
    CHECK_EQ(out->shape_view().elem_cnt(), elem_cnt);
    CHECK_EQ(in->data_type(), out->data_type());
    if (elem_cnt > 0) {
      std::unique_ptr<ep::primitive::Memcpy> primitive =
          ep::primitive::NewPrimitive<ep::primitive::MemcpyFactory>(
              ctx->stream()->device_type(), ep::primitive::MemcpyKind::kDtoD);
      CHECK(primitive) << "Can not create Memcpy primitive for device type "
                       << ctx->stream()->device_type();
      primitive->Launch(ctx->stream(), out->mut_dptr(), in->dptr(),
                        elem_cnt * GetSizeOfDataType(in->data_type()));
    }
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_COPY_DATA_CONTENT_KERNEL(op_type_name)                              \
  REGISTER_USER_KERNEL(op_type_name)                                                 \
      .SetCreateFn<CopyDataContentKernel>()                                          \
      .SetInplaceProposalFn(                                                         \
          [](const user_op::InferContext&,                                           \
             const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> { \
            OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, false));       \
            return Maybe<void>::Ok();                                                \
          });

REGISTER_COPY_DATA_CONTENT_KERNEL("squeeze");
REGISTER_COPY_DATA_CONTENT_KERNEL("reshape_like");
REGISTER_COPY_DATA_CONTENT_KERNEL("expand_dims");
REGISTER_COPY_DATA_CONTENT_KERNEL("reshape");
REGISTER_COPY_DATA_CONTENT_KERNEL("amp_white_identity");
REGISTER_COPY_DATA_CONTENT_KERNEL("amp_black_identity");
REGISTER_COPY_DATA_CONTENT_KERNEL("identity");
REGISTER_COPY_DATA_CONTENT_KERNEL("identity_buffer");
REGISTER_COPY_DATA_CONTENT_KERNEL("parallel_cast");
REGISTER_COPY_DATA_CONTENT_KERNEL("hierarchical_parallel_cast");
REGISTER_COPY_DATA_CONTENT_KERNEL("hierarchical_parallel_cast_like");
REGISTER_COPY_DATA_CONTENT_KERNEL("pinned_identity");
REGISTER_COPY_DATA_CONTENT_KERNEL("depend");

}  // namespace

}  // namespace oneflow
