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
#include "oneflow/core/embedding/cuda_in_memory_key_value_store.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/ep/include/primitive/memcpy.h"
#include "oneflow/core/embedding/embedding_manager.h"

namespace oneflow {

namespace {}  // namespace

template<typename IDX>
class EmbeddingLookupKernel final : public user_op::OpKernel {
 public:
  EmbeddingLookupKernel() = default;
  ~EmbeddingLookupKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    LOG(ERROR) << "EmbeddingLookupKernel";
    embedding::KeyValueStore* store = Global<EmbeddingMgr>::Get()->GetKeyValueStore(
        "MyEmbeddingTest", ctx->parallel_ctx().parallel_id());

    cudaStream_t cuda_stream = ctx->stream()->As<ep::CudaStream>()->cuda_stream();
    const user_op::Tensor* num_unique_ids = ctx->Tensor4ArgNameAndIndex("num_unique_ids", 0);
    const user_op::Tensor* unique_ids = ctx->Tensor4ArgNameAndIndex("unique_ids", 0);
    const user_op::Tensor* context = ctx->Tensor4ArgNameAndIndex("context", 0);
    user_op::Tensor* embeddings = ctx->Tensor4ArgNameAndIndex("embeddings", 0);
    IDX* host_num_keys;
    OF_CUDA_CHECK(cudaMallocHost(&host_num_keys, 1 * sizeof(IDX)));
    std::unique_ptr<ep::primitive::Memcpy> copyd2h_primitive =
        ep::primitive::NewPrimitive<ep::primitive::MemcpyFactory>(DeviceType::kCUDA,
                                                                  ep::primitive::MemcpyKind::kDtoH);
    CHECK(copyd2h_primitive);
    copyd2h_primitive->Launch(ctx->stream(), host_num_keys, num_unique_ids->dptr(), sizeof(IDX));
    // store->Lookup(stream, *host_num_keys, num_unique_ids->dptr<K>(), context->dptr<K>(),
    //          embeddings->dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_EMBEDDING_LOOKUP_KERNEL(dtype)                   \
  REGISTER_USER_KERNEL("embedding_lookup")                             \
      .SetCreateFn<EmbeddingLookupKernel<dtype>>()                     \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("unique_ids", 0) == GetDataType<dtype>::value));

REGISTER_CUDA_EMBEDDING_LOOKUP_KERNEL(int32_t)
REGISTER_CUDA_EMBEDDING_LOOKUP_KERNEL(int64_t)

template<typename T>
class EmbeddingUpdateKernel final : public user_op::OpKernel {
 public:
  EmbeddingUpdateKernel() = default;
  ~EmbeddingUpdateKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    LOG(ERROR) << "EmbeddingUpdateKernel";
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_EMBEDDING_UPDATE_KERNEL(dtype)                   \
  REGISTER_USER_KERNEL("sgd_embedding_update")                         \
      .SetCreateFn<EmbeddingUpdateKernel<dtype>>()                     \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("unique_ids", 0) == GetDataType<dtype>::value));

REGISTER_CUDA_EMBEDDING_UPDATE_KERNEL(int32_t)
REGISTER_CUDA_EMBEDDING_UPDATE_KERNEL(int64_t)

}  // namespace oneflow
