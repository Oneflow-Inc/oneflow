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
#include "oneflow/user/kernels/cumsum_kernel_util.h"

namespace oneflow {
namespace user_op {

template<DeviceType device_type, typename IN_T>
class CumsumKernel final : public user_op::OpKernel {
 public:
  CumsumKernel() = default;
  ~CumsumKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    Memcpy<device_type>(ctx->device_ctx(), out->mut_dptr<IN_T>(), in->dptr<IN_T>(),
                        in->shape().elem_cnt() * sizeof(IN_T));
    int32_t axis = ctx->Attr<int32_t>("axis");
    const bool exclusive = ctx->Attr<bool>("exclusive");
    const bool reverse = ctx->Attr<bool>("reverse");
    const int32_t elem_cnt = in->shape().elem_cnt();
    const int32_t instance_size = in->shape().At(axis);
    const int32_t instance_num = elem_cnt / instance_size;
    int32_t post = 1;
    FOR_RANGE(int32_t, i, axis + 1, in->shape().NumAxes()) { post *= in->shape().At(i); }
    const IN_T* in_ptr = in->dptr<IN_T>();
    IN_T* out_ptr = out->mut_dptr<IN_T>();

    CumsumFunctor<device_type, IN_T>()(ctx->device_ctx(), instance_num, instance_size, post,
                                       exclusive, reverse, in_ptr, out_ptr);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUMSUM_KERNEL(device, dtype)                                                \
  REGISTER_USER_KERNEL("cumsum").SetCreateFn<CumsumKernel<device, dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceTag() == device)                                                    \
      & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

#define REGISTER_CUMSUM_KERNELS_WITH_DEVICE(device) \
  REGISTER_CUMSUM_KERNEL(device, float)             \
  REGISTER_CUMSUM_KERNEL(device, double)            \
  REGISTER_CUMSUM_KERNEL(device, int32_t)           \
  REGISTER_CUMSUM_KERNEL(device, int32_t)

REGISTER_CUMSUM_KERNELS_WITH_DEVICE(DeviceType::kCPU);

#ifdef WITH_CUDA
REGISTER_CUMSUM_KERNELS_WITH_DEVICE(DeviceType::kGPU);
#endif  // WITH_CUDA

}  // namespace user_op
}  // namespace oneflow
