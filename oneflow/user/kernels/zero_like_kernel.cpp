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
#include "oneflow/core/ep/include/primitive/memset.h"

namespace oneflow {

namespace {

class ZeroLikeKernel final : public user_op::OpKernel {
 public:
  ZeroLikeKernel() = default;
  ~ZeroLikeKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int64_t elem_cnt = out->shape_view().elem_cnt();
    if (elem_cnt > 0) {
      std::unique_ptr<ep::primitive::Memset> primitive =
          ep::primitive::NewPrimitive<ep::primitive::MemsetFactory>(ctx->stream()->device_type());
      CHECK(primitive) << "Can not create Memset primitive for device type "
                       << ctx->stream()->device_type();
      primitive->Launch(ctx->stream(), out->mut_dptr(), 0,
                        elem_cnt * GetSizeOfDataType(out->data_type()));
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("zero_like").SetCreateFn<ZeroLikeKernel>();

}  // namespace

}  // namespace oneflow
