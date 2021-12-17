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

namespace {

template<typename T, typename IDX>
__global__ void SGDUpdateKernel(const int64_t embedding_size, const IDX* num_unique_ids,
                                const float* learning_rate, float learning_rate_val,
                                const T* model_diff, const T* model, T* updated_model) {
  if (learning_rate != nullptr) { learning_rate_val = *learning_rate; }
  const int64_t n = *num_unique_ids * embedding_size;
  CUDA_1D_KERNEL_LOOP(i, n) {
    const T model_val = model[i];
    updated_model[i] = model_val - learning_rate_val * model_diff[i];
  }
}

}  // namespace

template<typename K, typename IDX>
class EmbeddingPrefetchKernel final : public user_op::OpKernel {
 public:
  EmbeddingPrefetchKernel() = default;
  ~EmbeddingPrefetchKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    LOG(ERROR) << "EmbeddingPrefetchKernel";
    embedding::KeyValueStore* store = Global<EmbeddingMgr>::Get()->GetKeyValueStore(
        "MyEmbeddingTest", ctx->parallel_ctx().parallel_id());
    const user_op::Tensor* num_unique_ids = ctx->Tensor4ArgNameAndIndex("num_unique_ids", 0);
    const user_op::Tensor* unique_ids = ctx->Tensor4ArgNameAndIndex("unique_ids", 0);
    user_op::Tensor* context = ctx->Tensor4ArgNameAndIndex("context", 0);
    IDX* host_num_keys;
    OF_CUDA_CHECK(cudaMallocHost(&host_num_keys, 1 * sizeof(IDX)));
    std::unique_ptr<ep::primitive::Memcpy> copyd2h_primitive =
        ep::primitive::NewPrimitive<ep::primitive::MemcpyFactory>(DeviceType::kCUDA,
                                                                  ep::primitive::MemcpyKind::kDtoH);
    CHECK(copyd2h_primitive);
    copyd2h_primitive->Launch(ctx->stream(), host_num_keys, num_unique_ids->dptr(), sizeof(IDX));
    CHECK_JUST(ctx->stream()->Sync());
    uint32_t num_keys = *host_num_keys;
    store->Prefetch(ctx->stream(), num_keys, unique_ids->dptr(),
                    reinterpret_cast<uint64_t*>(context->mut_dptr()));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_EMBEDDING_PREFETCH_KERNEL(k_dtype, idx_dtype) \
  REGISTER_USER_KERNEL("embedding_prefetch")                        \
      .SetCreateFn<EmbeddingPrefetchKernel<k_dtype, idx_dtype>>()   \
      .SetIsMatchedHob(                                             \
          (user_op::HobDeviceType() == DeviceType::kCUDA)           \
          && (user_op::HobDataType("num_unique_ids", 0) == GetDataType<idx_dtype>::value));

REGISTER_CUDA_EMBEDDING_PREFETCH_KERNEL(int64_t, int32_t)

template<typename T, typename K, typename IDX>
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
    CHECK_JUST(ctx->stream()->Sync());
    store->Lookup(ctx->stream(), *host_num_keys, unique_ids->dptr(),
                  reinterpret_cast<const uint64_t*>(context->dptr()), embeddings->mut_dptr());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_EMBEDDING_LOOKUP_KERNEL(t_dtype, k_dtype, idx_dtype)                \
  REGISTER_USER_KERNEL("embedding_lookup")                                                \
      .SetCreateFn<EmbeddingLookupKernel<t_dtype, k_dtype, idx_dtype>>()                  \
      .SetIsMatchedHob(                                                                   \
          (user_op::HobDeviceType() == DeviceType::kCUDA)                                 \
          && (user_op::HobDataType("num_unique_ids", 0) == GetDataType<idx_dtype>::value) \
          && (user_op::HobDataType("unique_ids", 0) == GetDataType<k_dtype>::value)       \
          && (user_op::HobDataType("embeddings", 0) == GetDataType<t_dtype>::value));

REGISTER_CUDA_EMBEDDING_LOOKUP_KERNEL(float, int64_t, int32_t)

template<typename T, typename K, typename IDX>
class EmbeddingUpdateKernel final : public user_op::OpKernel {
 public:
  EmbeddingUpdateKernel() = default;
  ~EmbeddingUpdateKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    LOG(ERROR) << "EmbeddingUpdateKernel";
    embedding::KeyValueStore* store = Global<EmbeddingMgr>::Get()->GetKeyValueStore(
        "MyEmbeddingTest", ctx->parallel_ctx().parallel_id());

    const user_op::Tensor* num_unique_ids = ctx->Tensor4ArgNameAndIndex("num_unique_ids", 0);
    const user_op::Tensor* unique_ids = ctx->Tensor4ArgNameAndIndex("unique_ids", 0);
    const user_op::Tensor* context = ctx->Tensor4ArgNameAndIndex("context", 0);
    const user_op::Tensor* unique_embeddings = ctx->Tensor4ArgNameAndIndex("unique_embeddings", 0);
    const user_op::Tensor* embedding_diff = ctx->Tensor4ArgNameAndIndex("embedding_diff", 0);
    const int64_t embedding_size =
        unique_embeddings->shape().elem_cnt() / unique_ids->shape().elem_cnt();
    LOG(ERROR) << "embedding_size " << embedding_size;
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    T* update_unique_embeddings = reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>());

    IDX* host_num_keys;
    OF_CUDA_CHECK(cudaMallocHost(&host_num_keys, 1 * sizeof(IDX)));
    std::unique_ptr<ep::primitive::Memcpy> copyd2h_primitive =
        ep::primitive::NewPrimitive<ep::primitive::MemcpyFactory>(DeviceType::kCUDA,
                                                                  ep::primitive::MemcpyKind::kDtoH);
    CHECK(copyd2h_primitive);
    copyd2h_primitive->Launch(ctx->stream(), host_num_keys, num_unique_ids->dptr(), sizeof(IDX));
    CHECK_JUST(ctx->stream()->Sync());

    const float learning_rate_val = ctx->Attr<float>("learning_rate_val");
    const float* learning_rate_ptr = nullptr;
    if (ctx->has_input("learning_rate", 0)) {
      const user_op::Tensor* learning_rate = ctx->Tensor4ArgNameAndIndex("learning_rate", 0);
      learning_rate_ptr = learning_rate->dptr<float>();
    }
    // update kernel
    SGDUpdateKernel<T, IDX>
        <<<BlocksNum4ThreadsNum(embedding_diff->shape().elem_cnt()), kCudaThreadsNumPerBlock, 0,
           ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
            embedding_size, num_unique_ids->dptr<IDX>(), learning_rate_ptr, learning_rate_val,
            embedding_diff->dptr<T>(), unique_embeddings->dptr<T>(), update_unique_embeddings);

    store->Update(ctx->stream(), *host_num_keys, unique_ids->dptr(),
                  reinterpret_cast<const uint64_t*>(context->dptr()), update_unique_embeddings);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_EMBEDDING_UPDATE_KERNEL(t_dtype, k_dtype, idx_dtype)                  \
  REGISTER_USER_KERNEL("sgd_embedding_update")                                              \
      .SetCreateFn<EmbeddingUpdateKernel<t_dtype, k_dtype, idx_dtype>>()                    \
      .SetIsMatchedHob(                                                                     \
          (user_op::HobDeviceType() == DeviceType::kCUDA)                                   \
          && (user_op::HobDataType("num_unique_ids", 0) == GetDataType<idx_dtype>::value)   \
          && (user_op::HobDataType("unique_ids", 0) == GetDataType<k_dtype>::value)         \
          && (user_op::HobDataType("unique_embeddings", 0) == GetDataType<t_dtype>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                   \
        const user_op::TensorDesc& unique_embeddings =                                      \
            ctx->InputTensorDesc("unique_embeddings", 0);                                   \
        return unique_embeddings.shape().elem_cnt() * sizeof(t_dtype);                      \
      });

REGISTER_CUDA_EMBEDDING_UPDATE_KERNEL(float, int64_t, int32_t)

}  // namespace oneflow
