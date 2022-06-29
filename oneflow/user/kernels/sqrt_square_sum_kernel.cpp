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
#include <cstdint>
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/kernels/sqrt_square_sum_kernel_util.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/kernel/cuda_graph_support.h"

namespace oneflow {

namespace user_op {

int64_t getThreadNumBlocks(int64_t n) {
  int64_t num_blocks = 1;
#ifdef WITH_CUDA
  num_blocks = BlocksNum4ThreadsNum(n);
#endif
  return num_blocks;
}

template<DeviceType device_type, typename T>
class SqrtSquareSumKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  SqrtSquareSumKernel() = default;
  ~SqrtSquareSumKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* tmp = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);

    SqrtSquareSumKernelUtil<device_type, T>::SqrtSquareSum(ctx->stream(),
                                                           x->shape_view().elem_cnt(), x->dptr<T>(),
                                                           y->mut_dptr<T>(), tmp->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SQUARE_SUM_KERNEL(device, dtype)                                     \
  REGISTER_USER_KERNEL("sqrt_square_sum")                                             \
      .SetCreateFn<SqrtSquareSumKernel<device, OF_PP_PAIR_FIRST(dtype)>>()            \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                           \
                       && (user_op::HobDataType("y", 0) == OF_PP_PAIR_SECOND(dtype))) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t {                   \
        const auto& x_shape = ctx->InputTensorDesc("x", 0).shape();                   \
        const int32_t num_blocks = getThreadNumBlocks(x_shape.Count(0));              \
        int64_t tmp_buffer_size = num_blocks;                                         \
        return tmp_buffer_size * sizeof(OF_PP_PAIR_FIRST(dtype));                     \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SQUARE_SUM_KERNEL, DEVICE_TYPE_SEQ,
                                 FLOATING_DATA_TYPE_SEQ)

}  // namespace user_op

}  // namespace oneflow
