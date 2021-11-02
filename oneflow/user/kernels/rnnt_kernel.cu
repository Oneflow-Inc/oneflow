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
#include "oneflow/user/kernels/rnnt_kernel_gpu.h"
#include "oneflow/core/kernel/cuda_graph_support.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T>
class RNNTKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  RNNTKernel() = default;
  ~RNNTKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* acts = ctx->Tensor4ArgNameAndIndex("acts", 0);
    const user_op::Tensor* labels = ctx->Tensor4ArgNameAndIndex("labels", 0);
    const user_op::Tensor* act_lens = ctx->Tensor4ArgNameAndIndex("act_lens", 0);
    const user_op::Tensor* label_lens = ctx->Tensor4ArgNameAndIndex("label_lens", 0);
    const int32_t blank_label = ctx->Attr<int32_t>("blank_label");
    const int32_t num_threads = std::max(ctx->Attr<int32_t>("num_threads"),1);
    user_op::Tensor* costs = ctx->Tensor4ArgNameAndIndex("costs", 0);
    user_op::Tensor* grads = ctx->Tensor4ArgNameAndIndex("grads", 0);

    int32_t minibatch_size = acts->shape().At(0);
    int32_t maxT = acts->shape().At(1);
    int32_t maxU = acts->shape().At(2);
    int32_t alphabet_size = acts->shape().At(3);
    
    auto* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    T* gpu_workspace = tmp_buffer->mut_dptr<T>();

    GpuRNNT<T> rnnt(minibatch_size,
                    maxT,
                    maxU,
                    alphabet_size,
                    gpu_workspace,
                    blank_label,
                    num_threads,
                    ctx->device_ctx()->cuda_stream()
                    );
                    
    rnnt.cost_and_grad(acts->dptr<T>(), 
                       grads->mut_dptr<T>(),
                       costs->mut_dptr<T>(),
                       labels->dptr<int32_t>(),
                       label_lens->dptr<int32_t>(),
                       act_lens->dptr<int32_t>()
                       );

  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};


template<DeviceType device_type, typename T>
user_op::InferTmpSizeFn GenFwInferTmpSizeFn() {
  return [](user_op::InferContext* ctx) {
    const Shape& acts_shape = ctx->InputShape("acts", 0);
    int32_t minibatch_size = acts_shape.At(0); 
    int32_t maxT = acts_shape.At(1);
    int32_t maxU = acts_shape.At(2);
    
    return GetCudaAlignedSize(sizeof(T) * minibatch_size * (maxT * maxU * 3 + 2));
  };
}


#define REGISTER_RNNT_GPU_KERNEL(device, dtype)                                          \
  REGISTER_USER_KERNEL("RNNTloss")                                                       \
      .SetCreateFn<RNNTKernel<device, dtype>>()                                      \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                              \
                       & (user_op::HobDataType("acts", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn(GenFwInferTmpSizeFn<device, dtype>());

REGISTER_RNNT_GPU_KERNEL(DeviceType::kGPU, float)
REGISTER_RNNT_GPU_KERNEL(DeviceType::kGPU, double)

}

}  // namespace oneflow