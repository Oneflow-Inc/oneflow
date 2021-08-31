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
#include "oneflow/core/common/shape_vec.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/user/kernels/view_copy_kernel.h"

namespace oneflow {

template<>
void ViewCopyUtil<DeviceType::kCPU>::operator()() {
  if (contiguous_dim == -1) {
    std::memcpy(out_dptr, in_dptr, contiguous_block_size * dsize);
  } else {
    init_index_and_out_stride();

    while (true) {
      std::memcpy(out_dptr + out_offset * dsize, in_dptr + in_offset * dsize,
                  contiguous_block_size * dsize);

      if (next_index()) break;
    }
  }
}

namespace {

class ViewCopyKernel final : public user_op::OpKernel {
 public:
  ViewCopyKernel() = default;
  ~ViewCopyKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    const ShapeView& in_shape = in->shape();
    CHECK_EQ(out->shape(), in_shape);
    const DataType in_data_type = in->data_type();
    CHECK_EQ(out->data_type(), in_data_type);

    const auto& in_stride = ctx->Attr<std::vector<int64_t>>("stride");
    int64_t storage_offset = ctx->Attr<int64_t>("storage_offset");

    const size_t dsize = GetSizeOfDataType(in_data_type);
    const char* in_dptr = static_cast<const char*>(in->raw_dptr()) + storage_offset * dsize;
    char* out_dptr = static_cast<char*>(out->mut_raw_dptr());

    if (in->mem_case().has_host_mem() && out->mem_case().has_host_mem()) {
      ViewCopyUtil<kCPU>(ctx->device_ctx(), in_shape, dsize, in_stride, in_dptr, out_dptr)();
    } else {
#ifdef WITH_CUDA
      ViewCopyUtil<kGPU>(ctx->device_ctx(), in_shape, dsize, in_stride, in_dptr, out_dptr)();
#else
      UNIMPLEMENTED();
#endif  // WITH_CUDA
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("view_copy").SetCreateFn<ViewCopyKernel>().SetIsMatchedHob(user_op::HobTrue());

}  // namespace
}  // namespace oneflow
