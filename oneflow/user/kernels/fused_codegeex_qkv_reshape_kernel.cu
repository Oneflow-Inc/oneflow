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
#include <cassert>
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace {

template<typename T, int pack_size>
struct alignas(sizeof(T) * pack_size) Packed {
  __device__ Packed(T val) {
#pragma unroll
    for (int i = 0; i < pack_size; i++) { elem[i] = val; }
  }
  __device__ Packed() {
    // do nothing
  }
  union {
    T elem[pack_size];
  };
  __device__ void operator=(Packed<T, pack_size> packA) {
#pragma unroll
    for (int i = 0; i < pack_size; i++) { elem[i] = packA.elem[i]; }
  }
};

// [seq_length, batch_size, hidden_size] -> [seq_length, batch_size, head_num, size_per_head]
template<typename T, int pack_size>
__global__ void batch_reshape_for_qkv(const int n, const T* query, const T* key, const T* value,
                                      T* new_query, T* new_key, T* new_value) {
  const auto* query_pack_ptr = reinterpret_cast<const Packed<T, pack_size>*>(query);
  const auto* key_pack_ptr = reinterpret_cast<const Packed<T, pack_size>*>(key);
  const auto* value_pack_ptr = reinterpret_cast<const Packed<T, pack_size>*>(value);
  auto* new_query_pack_ptr = reinterpret_cast<Packed<T, pack_size>*>(new_query);
  auto* new_key_pack_ptr = reinterpret_cast<Packed<T, pack_size>*>(new_key);
  auto* new_value_pack_ptr = reinterpret_cast<Packed<T, pack_size>*>(new_value);
  assert(n % pack_size == 0);
  CUDA_1D_KERNEL_LOOP(i, n) {
    Packed<T, pack_size> query_pack = query_pack_ptr[i];
    Packed<T, pack_size> key_pack = key_pack_ptr[i];
    Packed<T, pack_size> value_pack = value_pack_ptr[i];
    new_query_pack_ptr[i] = query_pack;
    new_key_pack_ptr[i] = key_pack;
    new_value_pack_ptr[i] = value_pack;
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

    const int32_t n = query->shape_view().elem_cnt();
    if (n % 4 == 0) {
      RUN_CUDA_KERNEL((batch_reshape_for_qkv<T, 4>), ctx->stream(), n / 4, n / 4, query->dptr<T>(),
                      key->dptr<T>(), value->dptr<T>(), new_query->mut_dptr<T>(),
                      new_key->mut_dptr<T>(), new_value->mut_dptr<T>());
    } else if (n % 2 == 0) {
      RUN_CUDA_KERNEL((batch_reshape_for_qkv<T, 2>), ctx->stream(), n / 2, n / 2, query->dptr<T>(),
                      key->dptr<T>(), value->dptr<T>(), new_query->mut_dptr<T>(),
                      new_key->mut_dptr<T>(), new_value->mut_dptr<T>());
    } else {
      RUN_CUDA_KERNEL((batch_reshape_for_qkv<T, 1>), ctx->stream(), n, n, query->dptr<T>(),
                      key->dptr<T>(), value->dptr<T>(), new_query->mut_dptr<T>(),
                      new_key->mut_dptr<T>(), new_value->mut_dptr<T>());
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_CODEGEEX_QKV_RESHAPE_CUDA_KERNEL(dtype)         \
  REGISTER_USER_KERNEL("fused_codegeex_qkv_reshape")                   \
      .SetCreateFn<FusedCodegeexQkvReshapeGpuKernel<dtype>>()          \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("query", 0) == GetDataType<dtype>::value));

REGISTER_FUSED_CODEGEEX_QKV_RESHAPE_CUDA_KERNEL(float)
REGISTER_FUSED_CODEGEEX_QKV_RESHAPE_CUDA_KERNEL(half)
REGISTER_FUSED_CODEGEEX_QKV_RESHAPE_CUDA_KERNEL(double)

}  // namespace oneflow
