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
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    std::vector<int32_t> axis = ctx->Attr<std::vector<int32_t>>("axis");
    NpuCommand npu_command;
    std::vector<int64_t> shape_desc = {static_cast<int64_t>(axis.size())};
    CHECK_EQ(tmp_buffer->shape_view().elem_cnt(), sizeof(int));
    std::string key = "ReduceSumLike" + ShapeToString(axis);
    if(!const_tensor_map.count(key)) const_tensor_map[key] = axis;
    if(!shape_map.count(key)) shape_map[key] = shape_desc;
    // AclTensorWrapper wrap(tmp_buffer->mut_dptr<void>(), ACL_INT32, shape_desc.size(), shape_desc.data(), 
    //                       ACL_FORMAT_ND, sizeof(int32_t), axis.data(), key);
    npu_command.OpName("ReduceSum")
                .Input(x)
                .Input(key, axis.size(), ACL_INT32)
                .Output(y)
                .Attr("keep_dims",false)
                .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
                .Check();
    npu_command.Run()
               .Realease();
    //OF_NPU_CHECK(aclrtSynchronizeStream(ctx->stream()->As<ep::NpuStream>()->npu_stream()));  
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};


REGISTER_USER_KERNEL("reduce_sum_like")
    .SetCreateFn<ReduceSumLikeNpuKernel<DeviceType::kNPU>>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kNPU)
                     && (user_op::HobDataType("y", 0) == GetDataType<float16>::value))
    .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t{                              
        const auto& x = ctx->InputTensorDesc("x", 0);                                   
        size_t tmp_size = 0;                                                                  
        int shape_size =  sizeof(int);                                     
        tmp_size += shape_size;                                                               
        return tmp_size;                                                                      
    });   
REGISTER_USER_KERNEL("reduce_sum_like")
    .SetCreateFn<ReduceSumLikeNpuKernel<DeviceType::kNPU>>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kNPU)
                     && (user_op::HobDataType("y", 0) == GetDataType<float>::value))
    .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t{                              
        const auto& x = ctx->InputTensorDesc("x", 0);                                   
        size_t tmp_size = 0;                                                                  
        int shape_size = sizeof(int);                                   
        tmp_size += shape_size;                                                               
        return tmp_size;                                                                      
    });  
}  // namespace oneflow
