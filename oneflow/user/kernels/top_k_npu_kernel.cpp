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
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/user/ops/npu_command.h"

namespace oneflow {

template<typename T>
class TopKNpuKernel final : public user_op::OpKernel {
 public:
  TopKNpuKernel() = default;
  ~TopKNpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* indice = ctx->Tensor4ArgNameAndIndex("indice", 0);
    if (in->shape_view().elem_cnt() == 0) { return; }
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const int64_t instance_size = in->shape_view().At(in->shape_view().NumAxes() - 1);
    const int64_t instance_num = in->shape_view().elem_cnt() / instance_size;
    
    const int64_t k = std::min(static_cast<int64_t>(ctx->Attr<int32_t>("k")), instance_size);
    std::vector<int64_t> k_shape ={1};
    std::vector<int> k_vec = {static_cast<int>(k)};
    CHECK_EQ(tmp_buffer->shape_view().elem_cnt(), sizeof(int));
    std::string key = "TopKNpu" + ShapeToString(k_vec);
    if(!const_tensor_map.count(key))  const_tensor_map[key] = k_vec;
    if(!shape_map.count(key)) shape_map[key] = k_shape;
    // AclTensorWrapper wrap(tmp_buffer->mut_dptr<void>(), ACL_INT32, k_shape.size(), k_shape.data(), ACL_FORMAT_ND,
    //                         sizeof(int), k_vec.data(), key);
    int64_t dim = static_cast<int64_t>(ctx->Attr<int32_t>("dim"));
    bool largest = ctx->Attr<bool>("largest");
    bool sorted = ctx->Attr<bool>("sorted");
    NpuCommand npu_command;
    npu_command.OpName("TopKV2")
               .Input(in)
               .Input(key, k_vec.size(), ACL_INT32)
               .Attr("sorted",sorted)
               .Attr("dim", dim)
               .Attr("largest", largest)
               .Output(out)
               .Output(indice)
               .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
               .Check();
    npu_command.Run()
               .Realease();
    //OF_NPU_CHECK(aclrtSynchronizeStream(ctx->stream()->As<ep::NpuStream>()->npu_stream()));  
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_NPU_TOP_K_KERNEL(dtype)                                                \
  REGISTER_USER_KERNEL("top_k_npu")                                                     \
      .SetCreateFn<TopKNpuKernel<dtype>>()                                              \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kNPU)                   \
                       && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                               \
        return sizeof(int); \
      });

REGISTER_NPU_TOP_K_KERNEL(float)
REGISTER_NPU_TOP_K_KERNEL(float16)
REGISTER_NPU_TOP_K_KERNEL(double)
REGISTER_NPU_TOP_K_KERNEL(int8_t)
REGISTER_NPU_TOP_K_KERNEL(uint8_t)
REGISTER_NPU_TOP_K_KERNEL(int32_t)
REGISTER_NPU_TOP_K_KERNEL(int64_t)

}  // namespace oneflow
