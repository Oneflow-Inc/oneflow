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

namespace oneflow {

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

    const auto& stride = ctx->Attr<std::vector<int64_t>>("stride");
    int64_t storage_offset = ctx->Attr<int64_t>("storage_offset");

    const size_t dsize = GetSizeOfDataType(in_data_type);
    const char* in_dptr = static_cast<const char*>(in->raw_dptr()) + storage_offset * dsize;
    char* out_dptr = static_cast<char*>(out->mut_raw_dptr());

    int64_t contiguous_block_size = 1;
    int64_t contiguous_dim = in_shape.NumAxes() - 1;
    for (; contiguous_dim != -1; --contiguous_dim) {
      if (contiguous_block_size == stride[contiguous_dim]) {
        contiguous_block_size *= in_shape.At(contiguous_dim);
      } else {
        break;
      }
    }

    if (contiguous_dim == -1) {
      AutoMemcpy(ctx->device_ctx(), out_dptr, in_dptr, contiguous_block_size * dsize,
                 out->mem_case(), in->mem_case());
    } else {
      StrideVector out_stride(in_shape.NumAxes());

      int64_t sum = 1;
      for (int64_t i = out_stride.size() - 1; i != -1; --i) {
        out_stride[i] = sum;
        sum *= in_shape.At(i);
      }

      DimVector index{contiguous_dim + 1, 0};
      int64_t in_offset = 0, out_offset = 0;

      bool not_overflow = true;
      while (not_overflow) {
        AutoMemcpy(ctx->device_ctx(), out_dptr + out_offset * dsize, in_dptr + in_offset * dsize,
                   contiguous_block_size * dsize, out->mem_case(), in->mem_case());

        int64_t i = contiguous_dim;
        for (; i != -1; --i) {
          if (index[i] == in_shape.At(i)) {
            in_offset -= stride[i] * index[i];
            out_offset -= out_stride[i] * index[i];
            index[i] = 0;
          } else {
            index[i]++;
            in_offset += stride[i];
            out_offset += out_stride[i];
            break;
          }
        }
        if (i == -1) { not_overflow = false; }
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("view_copy").SetCreateFn<ViewCopyKernel>().SetIsMatchedHob(user_op::HobTrue());

}  // namespace
}  // namespace oneflow
