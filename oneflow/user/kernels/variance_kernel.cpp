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
#include "oneflow/core/ndarray/ndarray_reduce.h"
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/user/kernels/variance_kernel_util.h"

namespace oneflow {
namespace user_op {

template<DeviceType device_type, typename T>
class VarKernel final : public user_op::OpKernel {
 public:
  VarKernel() = default;
  ~VarKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* input = ctx->Tensor4ArgNameAndIndex("input", 0);
    user_op::Tensor* output = ctx->Tensor4ArgNameAndIndex("output", 0);
    const bool unbiased = ctx->Attr<bool>("unbiased");
    const T* in_ptr = input->dptr<T>();
    T* out_ptr = output->mut_dptr<T>();
    const std::vector<int32_t> axis = ctx->Attr<std::vector<int32_t>>("dim");
    T* tmp_buffer_ptr = axis.size() == input->shape().NumAxes()
                            ? ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0)->mut_dptr<T>()
                            : nullptr;
    VarParamHelper param_helper(input->shape(), axis, unbiased);
    VarFunctor<device_type, T>()(ctx->stream(), in_ptr, out_ptr, tmp_buffer_ptr,
                                 param_helper.param);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

size_t InferTmpBufferSize(user_op::InferContext* ctx) {
  const TensorDesc& input = ctx->InputTensorDesc("input", 0);
  const Shape& input_shape = input.shape();
  const std::vector<int32_t> axis = ctx->Attr<std::vector<int32_t>>("dim");
  if (axis.size() == input_shape.NumAxes()) {
    return static_cast<size_t>(std::ceil(std::sqrt(input.shape().elem_cnt())))
           * GetSizeOfDataType(input.data_type()) * 3;
  }
  return 0;
}

#define REGISTER_VAR_KERNEL(device, dtype)                                                    \
  REGISTER_USER_KERNEL("var")                                                                 \
      .SetCreateFn<VarKernel<device, dtype>>()                                                \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                                   \
                       && (user_op::HobAttr<DataType>("dtype") == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn(InferTmpBufferSize);

#define REGISTER_VAR_KERNELS_WITH_DEVICE(device) \
  REGISTER_VAR_KERNEL(device, float)             \
  REGISTER_VAR_KERNEL(device, double)

REGISTER_VAR_KERNELS_WITH_DEVICE(DeviceType::kCPU)
#ifdef WITH_CUDA
REGISTER_VAR_KERNELS_WITH_DEVICE(DeviceType::kCUDA)
#endif

#undef REGISTER_VAR_KERNELS_WITH_DEVICE
#undef REGISTER_VAR_KERNEL

template<DeviceType device_type, typename T>
class VarGradKernel final : public user_op::OpKernel {
 public:
  VarGradKernel() = default;
  ~VarGradKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    // TODO(liufengwei): Kernel implementation replaces functional::xx
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_VAR_GRAD_KERNEL(device, dtype)             \
  REGISTER_USER_KERNEL("var_grad")                          \
      .SetCreateFn<VarGradKernel<device, dtype>>()          \
      .SetIsMatchedHob((user_op::HobDeviceType() == device) \
                       && (user_op::HobAttr<DataType>("dtype") == GetDataType<dtype>::value));

#define REGISTER_VAR_GRAD_KERNELS_WITH_DEVICE(device) \
  REGISTER_VAR_GRAD_KERNEL(device, float)             \
  REGISTER_VAR_GRAD_KERNEL(device, double)

REGISTER_VAR_GRAD_KERNELS_WITH_DEVICE(DeviceType::kCPU)
#ifdef WITH_CUDA
REGISTER_VAR_GRAD_KERNELS_WITH_DEVICE(DeviceType::kCUDA)
#endif

#undef REGISTER_VAR_GRAD_KERNELS_WITH_DEVICE
#undef REGISTER_VAR_GRAD_KERNEL

}  // namespace user_op
}  // namespace oneflow
