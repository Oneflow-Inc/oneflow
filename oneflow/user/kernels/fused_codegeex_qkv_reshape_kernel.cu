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

namespace oneflow {

namespace {

// [seq_length, batch_size, hidden_size] -> [seq_length, batch_size, head_num, size_per_head]
template<typename T>
__global__ void batch_reshape_for_qkv(const int n, const T* query, const T* key, const T* value, T* new_query, T* new_key, T* new_value, const int seq_length, const int batch_size, const int hidden_size){
   CUDA_1D_KERNEL_LOOP(i, n) {
    const int i_div_hidden_size = i / hidden_size;
    const int seq_id = i_div_hidden_size / batch_size;
    const int batch_id = i_div_hidden_size - (i_div_hidden_size / batch_size) * batch_size;
    const int hidden_size_id = i - (i / hidden_size) * hidden_size;
    const int src_id = seq_id * batch_size * hidden_size + batch_id * hidden_size + hidden_size_id;
    new_query[i] = query[src_id];
    new_key[i] = key[src_id];
    new_value[i] = value[src_id];
   }
}

};  // namespace

template<typename T>
class FusedCodegeexQkvReshapeGpuKernel final : public user_op::OpKernel {
 public:
  FusedCodegeexQkvReshapeGpuKernel() = default;
  ~FusedCodegeexQkvReshapeGpuKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    // [seq_length, batch_size, hidden_size] -> [seq_length, batch_size, head_num, size_per_head]
    const user_op::Tensor* query = ctx->Tensor4ArgNameAndIndex("query", 0);
    const user_op::Tensor* key = ctx->Tensor4ArgNameAndIndex("key", 0);
    const user_op::Tensor* value = ctx->Tensor4ArgNameAndIndex("value", 0);

    user_op::Tensor* new_query = ctx->Tensor4ArgNameAndIndex("new_query", 0);
    user_op::Tensor* new_key = ctx->Tensor4ArgNameAndIndex("new_key", 0);
    user_op::Tensor* new_value = ctx->Tensor4ArgNameAndIndex("new_value", 0);
    const int seq_length = query->shape_view().At(0);
    const int batch_size = query->shape_view().At(1);
    const int hidden_size = query->shape_view().At(2);

    const int32_t n = query->shape_view().elem_cnt();
    RUN_CUDA_KERNEL((batch_reshape_for_qkv<T>), ctx->stream(), n, n,
                    query->dptr<T>(), key->dptr<T>(), value->dptr<T>(), new_query->mut_dptr<T>(),
                    new_key->mut_dptr<T>(), new_value->mut_dptr<T>(), seq_length, batch_size,
                    hidden_size);

  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_CODEGEEX_QKV_RESHAPE_CUDA_KERNEL(dtype)                      \
  REGISTER_USER_KERNEL("fused_codegeex_qkv_reshape")                                \
      .SetCreateFn<FusedCodegeexQkvReshapeGpuKernel<dtype>>()                      \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("query", 0) == GetDataType<dtype>::value));

REGISTER_FUSED_CODEGEEX_QKV_RESHAPE_CUDA_KERNEL(float)
REGISTER_FUSED_CODEGEEX_QKV_RESHAPE_CUDA_KERNEL(half)
REGISTER_FUSED_CODEGEEX_QKV_RESHAPE_CUDA_KERNEL(double)

}
