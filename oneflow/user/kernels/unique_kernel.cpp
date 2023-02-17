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
#include "oneflow/user/kernels/unique_kernel_util.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T, typename K>
class UniqueKernel final : public user_op::OpKernel {
 public:
  UniqueKernel() = default;
  ~UniqueKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* idx = ctx->Tensor4ArgNameAndIndex("idx", 0);
    user_op::Tensor* num_unique = ctx->Tensor4ArgNameAndIndex("num_unique", 0);
    user_op::Tensor* tmp = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const bool& sorted = ctx->Attr<bool>("sorted");
    void* tmp_ptr = tmp ? tmp->mut_dptr() : nullptr;
    int64_t tmp_size = tmp ? tmp->shape_view().elem_cnt() * GetSizeOfDataType(tmp->data_type()) : 0;
    UniqueKernelUtil<device_type, T, K>::Unique(
        ctx->stream(), x->shape_view().elem_cnt(), x->dptr<T>(), num_unique->mut_dptr<K>(),
        y->mut_dptr<T>(), idx->mut_dptr<K>(), tmp_ptr, tmp_size, sorted);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, typename T, typename K>
user_op::InferTmpSizeFn GenInferTmpSizeFn() {
  return [](user_op::InferContext* ctx) {
    const auto& x = ctx->InputTensorDesc("x", 0);
    int64_t workspace_size_in_bytes = 0;
    UniqueKernelUtil<device_type, T, K>::GetUniqueWorkspaceSizeInBytes(
        nullptr, x.shape().elem_cnt(), &workspace_size_in_bytes);

    return workspace_size_in_bytes;
  };
}

#define REGISTER_UNIQUE_KERNEL(device_type_v, data_type_pair, indices_type_pair)            \
  REGISTER_USER_KERNEL("unique")                                                            \
      .SetCreateFn<UniqueKernel<device_type_v, OF_PP_PAIR_FIRST(data_type_pair),            \
                                OF_PP_PAIR_FIRST(indices_type_pair)>>()                     \
      .SetIsMatchedHob(                                                                     \
          (user_op::HobDeviceType() == device_type_v)                                       \
          && (user_op::HobDataType("x", 0) == OF_PP_PAIR_SECOND(data_type_pair))            \
          && (user_op::HobDataType("idx", 0) == OF_PP_PAIR_SECOND(indices_type_pair)))      \
      .SetInferTmpSizeFn(GenInferTmpSizeFn<device_type_v, OF_PP_PAIR_FIRST(data_type_pair), \
                                           OF_PP_PAIR_FIRST(indices_type_pair)>());

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_UNIQUE_KERNEL, DEVICE_TYPE_SEQ, ARITHMETIC_DATA_TYPE_SEQ,
                                 INDEX_DATA_TYPE_SEQ)

}  // namespace

}  // namespace oneflow
