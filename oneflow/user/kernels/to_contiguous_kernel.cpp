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
#include "oneflow/user/kernels/to_contiguous_kernel.h"

namespace oneflow {

template<typename T>
struct ToContiguousUtil<DeviceType::kCPU, T> : ToContiguousUtilBase {
  using ToContiguousUtilBase::ToContiguousUtilBase;

  static constexpr size_t dsize = sizeof(T);

  void operator()() {
    if (contiguous_dim == -1) {
      std::memcpy(out_dptr, in_dptr, contiguous_block_size * dsize);
    } else {
      init_index();
      init_out_stride();

      while (true) {
        std::memcpy(out_dptr + out_offset * dsize, in_dptr + in_offset * dsize,
                    contiguous_block_size * dsize);

        if (next_index()) break;
      }
    }
  }
};

namespace {

template<DeviceType device_type, typename T>
class ToContiguousKernel final : public user_op::OpKernel {
 public:
  ToContiguousKernel() = default;
  ~ToContiguousKernel() override = default;

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

    const char* in_dptr = static_cast<const char*>(in->raw_dptr()) + storage_offset * sizeof(T);
    char* out_dptr = static_cast<char*>(out->mut_raw_dptr());

    ToContiguousUtil<device_type, T>(ctx->device_ctx(), in_shape, in_stride, in_dptr, out_dptr)();
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_TO_CONTIGUOUS_KERNEL(device_type, T)           \
  REGISTER_USER_KERNEL("to_contiguous")                         \
      .SetCreateFn<ToContiguousKernel<device_type, T>>()        \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device_type) \
                       & (user_op::HobDataType("in", 0) == GetDataType<T>::value));

#define REGISTER_TO_CONTIGUOUS_CPU_KERNEL(T) REGISTER_TO_CONTIGUOUS_KERNEL(DeviceType::kCPU, T)
#define REGISTER_TO_CONTIGUOUS_GPU_KERNEL(T) REGISTER_TO_CONTIGUOUS_KERNEL(DeviceType::kGPU, T)

#define REGISTER_TO_CONTIGUOUS_KERNEL_FOR_CPU_TYPES \
  OF_PP_FOR_EACH_TUPLE(REGISTER_TO_CONTIGUOUS_CPU_KERNEL, TO_CONTIGUOUS_TYPES)

#define REGISTER_TO_CONTIGUOUS_KERNEL_FOR_GPU_TYPES       \
  OF_PP_FOR_EACH_TUPLE(REGISTER_TO_CONTIGUOUS_GPU_KERNEL, \
                       TO_CONTIGUOUS_TYPES TO_CONTIGUOUS_GPU_SPECIAL_TYPE)

REGISTER_TO_CONTIGUOUS_KERNEL_FOR_CPU_TYPES
#ifdef WITH_CUDA
REGISTER_TO_CONTIGUOUS_KERNEL_FOR_GPU_TYPES
#endif

}  // namespace
}  // namespace oneflow
