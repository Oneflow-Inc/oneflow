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
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/user/ops/npu_command.h"
namespace oneflow {

namespace {

template<typename T>
class BroadcastLikeNpuKernel final : public user_op::OpKernel {
 public:
  BroadcastLikeNpuKernel() = default;
  ~BroadcastLikeNpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    return ;
    user_op::Tensor* in_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* like_tensor = ctx->Tensor4ArgNameAndIndex("like", 0);
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const auto& axis = ctx->Attr<std::vector<int32_t>>("broadcast_axes");
    std::vector<int32_t> like_shape;
    for(size_t i = 0;i<like_tensor->shape().NumAxes();++i)
    {
        like_shape.push_back(like_tensor->shape().ptr()[i]);
    }
    std::vector<int64_t> shape_desc;
    shape_desc.push_back(like_shape.size());
    CHECK_EQ(tmp_buffer->shape().elem_cnt(), mulVector(shape_desc)*sizeof(int32_t));
    std::string key = "BroadcastLikeNpu" + ShapeToString(like_shape);
    if(!const_tensor_map.count(key))  const_tensor_map[key] = like_shape;
    AclTensorWrapper wrap(tmp_buffer->mut_dptr<void>(),
                          ACL_INT32,
                          shape_desc.size(),
                          shape_desc.data(),
                          ACL_FORMAT_ND,
                          mulVector(shape_desc)*sizeof(int32_t),
                          like_shape.data(),
                          key); // dck_caution_here const_wrap
    NpuCommand npu_command;
    npu_command.OpName("BroadcastTo")
               .Input(in_tensor,"channels_nd")
               .Input(wrap)
               .Output(out_tensor,"channels_nd")
               .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
               .Check();
    npu_command.Run()
               .Realease();
    //OF_NPU_CHECK(aclrtSynchronizeStream(ctx->stream()->As<ep::NpuStream>()->npu_stream()));
    //PrintResult(out_tensor);
    //std::cout<<"BroadcastTo over"<<std::endl;
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

#define REGISTER_BROADCAST_LIKE_NPU_KERNEL(dtype)                                               \
  REGISTER_USER_KERNEL("broadcast_like")                                                        \
      .SetCreateFn<BroadcastLikeNpuKernel<dtype>>()                                             \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kNPU)                           \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value))          \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t{                              \
          const auto& like = ctx->InputTensorDesc("like", 0);                                   \
          size_t tmp_size = 0;                                                                  \
          int shape_size = like.shape().NumAxes() * sizeof(int32_t);                            \
          tmp_size += shape_size;                                                               \
          return tmp_size;                                                                      \
      });   


REGISTER_BROADCAST_LIKE_NPU_KERNEL(float)
REGISTER_BROADCAST_LIKE_NPU_KERNEL(float16)
REGISTER_BROADCAST_LIKE_NPU_KERNEL(double)
REGISTER_BROADCAST_LIKE_NPU_KERNEL(bool)
REGISTER_BROADCAST_LIKE_NPU_KERNEL(int8_t)
REGISTER_BROADCAST_LIKE_NPU_KERNEL(int32_t)
REGISTER_BROADCAST_LIKE_NPU_KERNEL(int64_t)

}  // namespace oneflow
