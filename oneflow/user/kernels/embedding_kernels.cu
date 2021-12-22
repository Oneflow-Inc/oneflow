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
#include "oneflow/core/common/str_util.h"
namespace oneflow {

namespace {

void DumpToFile(ep::Stream* stream, std::string filename, int64_t parallel_id, size_t data_size,
                const void* ptr) {
  void* host_ptr;
  OF_CUDA_CHECK(cudaMallocHost(&host_ptr, data_size));
  std::unique_ptr<ep::primitive::Memcpy> copyd2h_primitive =
      ep::primitive::NewPrimitive<ep::primitive::MemcpyFactory>(DeviceType::kCUDA,
                                                                ep::primitive::MemcpyKind::kDtoH);
  CHECK(copyd2h_primitive);
  copyd2h_primitive->Launch(stream, host_ptr, ptr, data_size);
  CHECK_JUST(stream->Sync());
  std::ofstream dx_os;
  dx_os.open(StrCat("test/" + filename + "_", parallel_id));
  dx_os.write(reinterpret_cast<char*>(host_ptr), data_size);
  dx_os.close();
  OF_CUDA_CHECK(cudaFreeHost(host_ptr));
}

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

template<typename T, typename K>
__global__ void InitValueKernel(const int64_t embedding_size, const uint32_t* num_missing_keys,
                                const K* missing_keys, const uint32_t* missing_indices, T* values) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("num_store_missing_keys %d\n", static_cast<int>(*num_missing_keys));
  }
  for (int row = blockIdx.x; row < *num_missing_keys; row += gridDim.x) {
    const uint32_t index = missing_indices[row];
    for (int col = threadIdx.x; col < embedding_size; col += blockDim.x) {
      const int64_t offset = index * embedding_size + col;
      values[offset] = missing_keys[row];  // for debug, random init
    }
  }
}

template<typename T, typename K>
class PrefetchTmpBufferManager final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PrefetchTmpBufferManager);
  PrefetchTmpBufferManager(void* ptr, const int64_t num_keys, const int64_t embedding_size)
      : ptr_(ptr) {
    const size_t num_cache_missing_bytes = GetCudaAlignedSize(1 * sizeof(uint32_t));
    const size_t cache_missing_keys_bytes = GetCudaAlignedSize(num_keys * sizeof(K));
    const size_t cache_missing_indices_bytes = GetCudaAlignedSize(num_keys * sizeof(uint32_t));
    const size_t num_store_missing_bytes = GetCudaAlignedSize(1 * sizeof(uint32_t));
    const size_t store_missing_keys_bytes = GetCudaAlignedSize(num_keys * sizeof(K));
    const size_t store_missing_indices_bytes = GetCudaAlignedSize(num_keys * sizeof(uint32_t));
    const size_t store_values_bytes = GetCudaAlignedSize(num_keys * embedding_size * sizeof(T));
    const size_t num_cache_evited_bytes = GetCudaAlignedSize(1 * sizeof(uint32_t));
    const size_t cache_evited_keys_bytes = GetCudaAlignedSize(num_keys * sizeof(K));
    const size_t cache_evited_indices_bytes = GetCudaAlignedSize(num_keys * sizeof(uint32_t));
    const size_t cache_evited_values_bytes =
        GetCudaAlignedSize(num_keys * embedding_size * sizeof(T));

    num_cache_missing_offset_ = 0;
    cache_missing_keys_offset_ = num_cache_missing_offset_ + num_cache_missing_bytes;
    cache_missing_indices_offset_ = cache_missing_keys_offset_ + cache_missing_keys_bytes;
    num_store_missing_offset_ = cache_missing_indices_offset_ + cache_missing_indices_bytes;
    store_missing_keys_offset_ = num_store_missing_offset_ + num_store_missing_bytes;
    store_missing_indices_offset_ = store_missing_keys_offset_ + store_missing_keys_bytes;
    store_values_offset_ = store_missing_indices_offset_ + store_missing_indices_bytes;
    num_cache_evited_offset_ = store_values_offset_ + store_values_bytes;
    cache_evited_keys_offset_ = num_cache_evited_offset_ + num_cache_evited_bytes;
    cache_evited_indices_offset_ = cache_evited_keys_offset_ + cache_evited_keys_bytes;
    cache_evited_values_offset_ = cache_evited_indices_offset_ + cache_evited_indices_bytes;
    total_buffer_size_ = num_cache_missing_bytes + cache_missing_keys_bytes
                         + cache_missing_indices_bytes + num_store_missing_bytes
                         + store_missing_keys_bytes + store_missing_indices_bytes
                         + store_values_bytes + num_cache_evited_bytes + cache_evited_keys_bytes
                         + cache_evited_indices_bytes + cache_evited_values_bytes;
  }
  ~PrefetchTmpBufferManager() = default;

  size_t TotalBufferSize() const { return total_buffer_size_; }

  uint32_t* NumCacheMissingPtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(ptr_) + num_cache_missing_offset_);
  }
  K* CacheMissingKeysPtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<K*>(reinterpret_cast<char*>(ptr_) + cache_missing_keys_offset_);
  }
  uint32_t* CacheMissingIndicesPtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(ptr_)
                                       + cache_missing_indices_offset_);
  }
  uint32_t* NumStoreMissingPtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(ptr_) + num_store_missing_offset_);
  }
  K* StoreMissingKeysPtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<K*>(reinterpret_cast<char*>(ptr_) + store_missing_keys_offset_);
  }
  uint32_t* StoreMissingIndicesPtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(ptr_)
                                       + store_missing_indices_offset_);
  }
  T* StoreValuesPtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<T*>(reinterpret_cast<char*>(ptr_) + store_values_offset_);
  }
  uint32_t* NumCacheEvictedPtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(ptr_) + num_cache_evited_offset_);
  }
  K* CacheEvictedKeysPtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<K*>(reinterpret_cast<char*>(ptr_) + cache_evited_keys_offset_);
  }
  uint32_t* CacheEvictedIndicesPtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(ptr_)
                                       + cache_evited_indices_offset_);
  }
  T* CacheEvictedValuesPtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<T*>(reinterpret_cast<char*>(ptr_) + cache_evited_values_offset_);
  }

 private:
  size_t num_cache_missing_offset_;
  size_t cache_missing_keys_offset_;
  size_t cache_missing_indices_offset_;
  size_t num_store_missing_offset_;
  size_t store_missing_keys_offset_;
  size_t store_missing_indices_offset_;
  size_t store_values_offset_;
  size_t num_cache_evited_offset_;
  size_t cache_evited_keys_offset_;
  size_t cache_evited_indices_offset_;
  size_t cache_evited_values_offset_;
  size_t total_buffer_size_;
  void* ptr_;
};

class EmbeddingKernelState final : public user_op::OpKernelState {
 public:
  explicit EmbeddingKernelState(user_op::KernelInitContext* ctx) {
    OF_CUDA_CHECK(cudaMallocHost(&host_num_keys_, 1 * sizeof(int32_t)));  // TODO: int32_t->IDX
  }
  ~EmbeddingKernelState() { OF_CUDA_CHECK(cudaFreeHost(host_num_keys_)); }

  void* HostNumKeys() { return host_num_keys_; }

 private:
  void* host_num_keys_;
};

}  // namespace

template<typename T, typename K, typename IDX>
class EmbeddingPrefetchKernel final : public user_op::OpKernel {
 public:
  EmbeddingPrefetchKernel() = default;
  ~EmbeddingPrefetchKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<EmbeddingKernelState>(ctx);
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    auto* kernel_state = dynamic_cast<EmbeddingKernelState*>(state);
    CHECK(kernel_state != nullptr);
    embedding::Cache* cache =
        Global<EmbeddingMgr>::Get()->GetCache("MyEmbeddingTest", ctx->parallel_ctx().parallel_id());
    embedding::KeyValueStore* store = Global<EmbeddingMgr>::Get()->GetKeyValueStore(
        "MyEmbeddingTest", ctx->parallel_ctx().parallel_id());
    const user_op::Tensor* num_unique_ids = ctx->Tensor4ArgNameAndIndex("num_unique_ids", 0);
    const user_op::Tensor* unique_ids = ctx->Tensor4ArgNameAndIndex("unique_ids", 0);
    user_op::Tensor* context = ctx->Tensor4ArgNameAndIndex("context", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const int64_t embedding_size = 128;
    PrefetchTmpBufferManager<T, K> buffer_manager(tmp_buffer->mut_dptr(),
                                                  unique_ids->shape().elem_cnt(), embedding_size);
    uint32_t* host_num_keys = reinterpret_cast<uint32_t*>(kernel_state->HostNumKeys());
    std::unique_ptr<ep::primitive::Memcpy> copyd2h_primitive =
        ep::primitive::NewPrimitive<ep::primitive::MemcpyFactory>(DeviceType::kCUDA,
                                                                  ep::primitive::MemcpyKind::kDtoH);
    CHECK(copyd2h_primitive);
    copyd2h_primitive->Launch(ctx->stream(), host_num_keys, num_unique_ids->dptr(), sizeof(IDX));
    CHECK_JUST(ctx->stream()->Sync());
    uint32_t num_keys = *host_num_keys;
    LOG(ERROR) << ctx->parallel_ctx().parallel_id() << " num_unique_ids " << num_keys;

    cache->Test(ctx->stream(), num_keys, unique_ids->dptr(), buffer_manager.NumCacheMissingPtr(),
                buffer_manager.CacheMissingKeysPtr(), buffer_manager.CacheMissingIndicesPtr());
    DumpToFile(ctx->stream(), "CacheMissingKeysPtr", ctx->parallel_ctx().parallel_id(),
               num_keys * sizeof(K), buffer_manager.CacheMissingKeysPtr());
    copyd2h_primitive->Launch(ctx->stream(), host_num_keys, buffer_manager.NumCacheMissingPtr(),
                              sizeof(uint32_t));
    CHECK_JUST(ctx->stream()->Sync());
    uint32_t num_missing_keys = *host_num_keys;
    LOG(ERROR) << ctx->parallel_ctx().parallel_id() << " num_cache_missing_keys "
               << num_missing_keys;

    store->Get(ctx->stream(), num_missing_keys, buffer_manager.CacheMissingKeysPtr(),
               buffer_manager.StoreValuesPtr(), buffer_manager.NumStoreMissingPtr(),
               buffer_manager.StoreMissingKeysPtr(), buffer_manager.StoreMissingIndicesPtr(),
               reinterpret_cast<uint64_t*>(context->mut_dptr()));
    if (ParseBooleanFromEnv("DEBUG_SHUFFLE", false)) {
      DumpToFile(ctx->stream(), "NumStoreMissingPtr", ctx->parallel_ctx().parallel_id(),
                 1 * sizeof(uint32_t), buffer_manager.NumStoreMissingPtr());
      DumpToFile(ctx->stream(), "StoreMissingKeysPtr", ctx->parallel_ctx().parallel_id(),
                 num_keys * sizeof(K), buffer_manager.StoreMissingKeysPtr());
      DumpToFile(ctx->stream(), "StoreMissingIndicesPtr", ctx->parallel_ctx().parallel_id(),
                 num_keys * sizeof(K), buffer_manager.StoreMissingIndicesPtr());
      DumpToFile(ctx->stream(), "StoreValuesPtr", ctx->parallel_ctx().parallel_id(),
                 num_keys * embedding_size * sizeof(K), buffer_manager.StoreValuesPtr());
    }
    // init values
    InitValueKernel<T, K><<<BlocksNum4ThreadsNum(num_missing_keys), embedding_size, 0,
                            ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
        embedding_size, buffer_manager.NumStoreMissingPtr(), buffer_manager.StoreMissingKeysPtr(),
        buffer_manager.StoreMissingIndicesPtr(), buffer_manager.StoreValuesPtr());
    if (ParseBooleanFromEnv("DEBUG_SHUFFLE", false)) {
      DumpToFile(ctx->stream(), "StoreValuesPtrAfterInit", ctx->parallel_ctx().parallel_id(),
                 num_keys * sizeof(K), buffer_manager.StoreValuesPtr());
    }
    cache->Put(ctx->stream(), num_missing_keys, buffer_manager.CacheMissingKeysPtr(),
               buffer_manager.StoreValuesPtr(), buffer_manager.NumCacheEvictedPtr(),
               buffer_manager.CacheEvictedKeysPtr(), buffer_manager.CacheEvictedValuesPtr());

    copyd2h_primitive->Launch(ctx->stream(), host_num_keys, buffer_manager.NumCacheEvictedPtr(),
                              sizeof(uint32_t));
    CHECK_JUST(ctx->stream()->Sync());
    uint32_t num_evicted_keys = *host_num_keys;
    LOG(ERROR) << ctx->parallel_ctx().parallel_id() << " num_evicted_keys " << num_evicted_keys;
    store->Put(ctx->stream(), num_evicted_keys, buffer_manager.CacheEvictedKeysPtr(),
               buffer_manager.CacheEvictedValuesPtr(),
               reinterpret_cast<uint64_t*>(context->mut_dptr()));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_EMBEDDING_PREFETCH_KERNEL(t_dtype, k_dtype, idx_dtype)               \
  REGISTER_USER_KERNEL("embedding_prefetch")                                               \
      .SetCreateFn<EmbeddingPrefetchKernel<t_dtype, k_dtype, idx_dtype>>()                 \
      .SetIsMatchedHob(                                                                    \
          (user_op::HobDeviceType() == DeviceType::kCUDA)                                  \
          && (user_op::HobDataType("num_unique_ids", 0) == GetDataType<idx_dtype>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                  \
        const int64_t embedding_size = 128;                                                \
        const user_op::TensorDesc& unique_ids = ctx->InputTensorDesc("unique_ids", 0);     \
        PrefetchTmpBufferManager<t_dtype, k_dtype> buffer_manager(                         \
            nullptr, unique_ids.shape().elem_cnt(), embedding_size);                       \
        return buffer_manager.TotalBufferSize();                                           \
      });

REGISTER_CUDA_EMBEDDING_PREFETCH_KERNEL(float, int64_t, int32_t)

template<typename T, typename K, typename IDX>
class EmbeddingLookupKernel final : public user_op::OpKernel {
 public:
  EmbeddingLookupKernel() = default;
  ~EmbeddingLookupKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<EmbeddingKernelState>(ctx);
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    auto* kernel_state = dynamic_cast<EmbeddingKernelState*>(state);
    CHECK(kernel_state != nullptr);
    embedding::Cache* cache =
        Global<EmbeddingMgr>::Get()->GetCache("MyEmbeddingTest", ctx->parallel_ctx().parallel_id());
    embedding::KeyValueStore* store = Global<EmbeddingMgr>::Get()->GetKeyValueStore(
        "MyEmbeddingTest", ctx->parallel_ctx().parallel_id());

    const user_op::Tensor* num_unique_ids = ctx->Tensor4ArgNameAndIndex("num_unique_ids", 0);
    const user_op::Tensor* unique_ids = ctx->Tensor4ArgNameAndIndex("unique_ids", 0);
    const user_op::Tensor* context = ctx->Tensor4ArgNameAndIndex("context", 0);
    user_op::Tensor* embeddings = ctx->Tensor4ArgNameAndIndex("embeddings", 0);

    // just for debug, not need such tmp_buffer
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const int64_t embedding_size = 128;
    PrefetchTmpBufferManager<T, K> buffer_manager(tmp_buffer->mut_dptr(),
                                                  unique_ids->shape().elem_cnt(), embedding_size);

    uint32_t* host_num_keys = reinterpret_cast<uint32_t*>(kernel_state->HostNumKeys());
    std::unique_ptr<ep::primitive::Memcpy> copyd2h_primitive =
        ep::primitive::NewPrimitive<ep::primitive::MemcpyFactory>(DeviceType::kCUDA,
                                                                  ep::primitive::MemcpyKind::kDtoH);
    CHECK(copyd2h_primitive);
    copyd2h_primitive->Launch(ctx->stream(), host_num_keys, num_unique_ids->dptr(),
                              sizeof(uint32_t));
    CHECK_JUST(ctx->stream()->Sync());

    user_op::Tensor* out_context = ctx->Tensor4ArgNameAndIndex("out_context", 0);
    std::unique_ptr<ep::primitive::Memcpy> copyd2d_primitive =
        ep::primitive::NewPrimitive<ep::primitive::MemcpyFactory>(DeviceType::kCUDA,
                                                                  ep::primitive::MemcpyKind::kDtoD);
    CHECK(copyd2d_primitive);
    copyd2d_primitive->Launch(ctx->stream(), out_context->mut_dptr(), context->dptr(),
                              context->shape().elem_cnt() * sizeof(uint64_t));

    cache->Get(ctx->stream(), *host_num_keys, unique_ids->dptr(), embeddings->mut_dptr(),
               buffer_manager.NumCacheMissingPtr(), buffer_manager.CacheMissingKeysPtr(),
               buffer_manager.CacheMissingIndicesPtr());
    copyd2h_primitive->Launch(ctx->stream(), host_num_keys, buffer_manager.NumCacheMissingPtr(),
                              sizeof(uint32_t));
    CHECK_JUST(ctx->stream()->Sync());
    LOG(ERROR) << "num_cache_missing" << *host_num_keys;
    if (*host_num_keys != 0) {
      store->Get(ctx->stream(), *host_num_keys, buffer_manager.CacheMissingKeysPtr(),
                 buffer_manager.StoreValuesPtr(), buffer_manager.NumStoreMissingPtr(),
                 buffer_manager.StoreMissingKeysPtr(), buffer_manager.StoreMissingIndicesPtr(),
                 reinterpret_cast<uint64_t*>(context->mut_dptr()));
      copyd2h_primitive->Launch(ctx->stream(), host_num_keys, buffer_manager.NumCacheMissingPtr(),
                              sizeof(uint32_t));
      CHECK_JUST(ctx->stream()->Sync());
      CHECK_GT(*host_num_keys, 0); //we think keys must be in cache or kv_store.
      //scatter

    }
    
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
          && (user_op::HobDataType("embeddings", 0) == GetDataType<t_dtype>::value))      \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                 \
        const int64_t embedding_size = 128;                                               \
        const user_op::TensorDesc& unique_ids = ctx->InputTensorDesc("unique_ids", 0);    \
        PrefetchTmpBufferManager<t_dtype, k_dtype> buffer_manager(                        \
            nullptr, unique_ids.shape().elem_cnt(), embedding_size);                      \
        return buffer_manager.TotalBufferSize();                                          \
      });

REGISTER_CUDA_EMBEDDING_LOOKUP_KERNEL(float, int64_t, int32_t)

template<typename T, typename K, typename IDX>
class EmbeddingUpdateKernel final : public user_op::OpKernel {
 public:
  EmbeddingUpdateKernel() = default;
  ~EmbeddingUpdateKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<EmbeddingKernelState>(ctx);
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    auto* kernel_state = dynamic_cast<EmbeddingKernelState*>(state);
    CHECK(kernel_state != nullptr);
    embedding::KeyValueStore* store = Global<EmbeddingMgr>::Get()->GetKeyValueStore(
        "MyEmbeddingTest", ctx->parallel_ctx().parallel_id());

    const user_op::Tensor* num_unique_ids = ctx->Tensor4ArgNameAndIndex("num_unique_ids", 0);
    const user_op::Tensor* unique_ids = ctx->Tensor4ArgNameAndIndex("unique_ids", 0);
    const user_op::Tensor* context = ctx->Tensor4ArgNameAndIndex("context", 0);
    const user_op::Tensor* unique_embeddings = ctx->Tensor4ArgNameAndIndex("unique_embeddings", 0);
    const user_op::Tensor* embedding_diff = ctx->Tensor4ArgNameAndIndex("embedding_diff", 0);
    const int64_t embedding_size =
        unique_embeddings->shape().elem_cnt() / unique_ids->shape().elem_cnt();
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    T* update_unique_embeddings = reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>());

    IDX* host_num_keys = reinterpret_cast<IDX*>(kernel_state->HostNumKeys());
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

    //    store->Update(ctx->stream(), *host_num_keys, unique_ids->dptr(),
    //                  reinterpret_cast<const uint64_t*>(context->dptr()),
    //                  update_unique_embeddings);
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
