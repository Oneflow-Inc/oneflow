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
#include "oneflow/core/ndarray/binary_func.h"
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/ep/include/primitive/cast.h"
#include "oneflow/core/ep/include/primitive/fill.h"
#include "oneflow/user/ops/npu_command.h"
namespace oneflow {


template<DeviceType device_type>
class ReduceSumLikeNpuKernel final : public user_op::OpKernel {
 public:
  ReduceSumLikeNpuKernel() = default;
  ~ReduceSumLikeNpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    std::vector<int32_t> axis = ctx->Attr<std::vector<int32_t>>("axis");
    std::cout<<"ReduceSumLikeNpuKernel tensor_x shape"<< x->shape().ToString()<<std::endl;
    std::cout<<"ReduceSumLikeNpuKernel tensor_y shape"<< y->shape().ToString()<<std::endl;
    VECTOR_PRINT(axis);
    NpuCommand npu_command;
    std::vector<int64_t> axis_shape = {static_cast<int64_t>(axis.size())};
    VECTOR_PRINT(axis_shape);
    AclTensorWrapper wrap(nullptr, ACL_INT32, axis_shape.size(), axis_shape.data(), 
                          ACL_FORMAT_ND, sizeof(int32_t), axis.data(), /**isConst**/true);
    npu_command.OpName("ReduceSum")
                .Input(x)
                .Input(wrap)
                .Output(y)
                .Attr("keep_dims",false)
                .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
                .Check();
    npu_command.Run();
    OF_NPU_CHECK(aclrtSynchronizeStream(ctx->stream()->As<ep::NpuStream>()->npu_stream()));   
    //PrintResult(y);
    // if (tensor_x->shape().elem_cnt() == 0) {
    //   if (tensor_y->shape().elem_cnt() != 0) {
    //     Memset<device_type>(
    //         ctx->stream(), tensor_y->mut_dptr<T>(), 0,
    //         tensor_y->shape().elem_cnt() * GetSizeOfDataType(tensor_y->data_type()));
    //   }
    //   return;
    // }
    // if (axis.empty()) {
    //   CHECK_EQ(tensor_x->shape(), tensor_y->shape());
    //   Memcpy<device_type>(ctx->stream(), tensor_y->mut_dptr(), tensor_x->dptr(),
    //                       tensor_x->shape().elem_cnt() * GetSizeOfDataType(tensor_x->data_type()));
    // } 

  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};


REGISTER_USER_KERNEL("reduce_sum_like")
    .SetCreateFn<ReduceSumLikeNpuKernel<DeviceType::kNPU>>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kNPU)
                     && (user_op::HobDataType("y", 0) == GetDataType<float16>::value));

REGISTER_USER_KERNEL("reduce_sum_like")
    .SetCreateFn<ReduceSumLikeNpuKernel<DeviceType::kNPU>>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kNPU)
                     && (user_op::HobDataType("y", 0) == GetDataType<float>::value));
}  // namespace oneflow
