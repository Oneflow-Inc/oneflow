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
#include "oneflow/core/embedding/key_value_store.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/ep/include/primitive/memcpy.h"
#include "oneflow/core/embedding/embedding_manager.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/user/kernels/random_mask_generator.h"
#include "oneflow/core/framework/random_generator_impl.h"
#include "oneflow/core/cuda/atomic.cuh"
#include "oneflow/core/embedding/embedding_options.h"
#include "oneflow/core/ep/include/primitive/copy_nd.h"
#include "oneflow/core/ep/include/primitive/cast.h"

namespace oneflow {

namespace {

constexpr size_t kMaxColumns = 128;

struct InitParam {
  int32_t num_columns = 0;
  embedding::EmbeddingInitializer initializer[kMaxColumns];
};

template<typename T, typename G, typename IDX>
__global__ void SGDUpdateKernel(const int64_t embedding_size, T scale, const IDX* num_unique_ids,
                                const float* learning_rate, const T* scale_by_ptr,
                                const int64_t* skip_if, const G* model_diff, const T* model,
                                T* updated_model) {
  if (skip_if != nullptr && *skip_if != 0) { return; }
  if (scale_by_ptr != nullptr) { scale /= *scale_by_ptr; }
  float learning_rate_val = *learning_rate;
  const int64_t n = *num_unique_ids * embedding_size;
  CUDA_1D_KERNEL_LOOP(i, n) {
    const T model_val = model[i];
    updated_model[i] = model_val - learning_rate_val * (scale * static_cast<T>(model_diff[i]));
  }
}

template<typename T, typename G, typename IDX>
__global__ void MomentumUpdateKernel(const int64_t line_size, const int64_t embedding_size, T scale,
                                     float beta, const IDX* num_unique_ids,
                                     const float* learning_rate, const T* scale_by_ptr,
                                     const int64_t* skip_if, const G* model_diff,
                                     const T* unique_values, T* updated_unique_values) {
  if (skip_if != nullptr && *skip_if != 0) { return; }
  if (scale_by_ptr != nullptr) { scale *= *scale_by_ptr; }
  float learning_rate_val = *learning_rate;
  const int64_t rows = *num_unique_ids;
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    const int64_t row_offset = row * line_size;
    for (int col = threadIdx.x; col < embedding_size; col += blockDim.x) {
      const int64_t offset = row_offset + col;
      const int64_t momentum_offset = row_offset + embedding_size + col;
      const T model_val = unique_values[offset];
      const T momentum = unique_values[momentum_offset];
      const T model_diff_val = scale * static_cast<T>(model_diff[offset]);
      const T next_momentum = beta * momentum - learning_rate_val * model_diff_val;
      const T next_model = model_val + next_momentum;
      updated_unique_values[offset] = next_model;
      updated_unique_values[momentum_offset] = next_momentum;
    }
  }
}

template<typename T, typename G, typename IDX>
__global__ void AdamUpdateKernel(const int64_t line_size, const int64_t embedding_size, T scale,
                                 float beta1, float beta2, float epsilon,
                                 const float* bias_correction1_ptr,
                                 const float* bias_correction2_ptr, const IDX* num_unique_ids,
                                 const float* learning_rate, const T* scale_by_ptr,
                                 const int64_t* skip_if, const G* model_diff,
                                 const T* unique_values, T* updated_unique_values) {
  if (skip_if != nullptr && *skip_if != 0) { return; }
  if (scale_by_ptr != nullptr) { scale *= *scale_by_ptr; }
  float learning_rate_val = *learning_rate;
  float bias_correction1_val = 1.0;
  float bias_correction2_val = 1.0;
  if (bias_correction1_ptr != nullptr) { bias_correction1_val = *bias_correction1_ptr; }
  if (bias_correction2_ptr != nullptr) { bias_correction2_val = *bias_correction2_ptr; }
  const int64_t rows = *num_unique_ids;
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    const int64_t row_offset = row * line_size;
    for (int col = threadIdx.x; col < embedding_size; col += blockDim.x) {
      const int64_t offset = row_offset + col;
      const int64_t m_offset = row_offset + embedding_size + col;
      const int64_t v_offset = row_offset + 2 * embedding_size + col;

      const T model_val = unique_values[offset];
      const T m = unique_values[m_offset];
      const T v = unique_values[v_offset];
      const T model_diff_value = scale * static_cast<T>(model_diff[offset]);
      const T next_m = beta1 * m + (1 - beta1) * model_diff_value;
      const T next_v = beta2 * v + (1 - beta2) * model_diff_value * model_diff_value;
      T denom = (sqrt(next_v) / sqrt(bias_correction2_val)) + epsilon;
      const T step_size = learning_rate_val / bias_correction1_val;
      updated_unique_values[offset] = model_val - step_size * (next_m / denom);
      updated_unique_values[m_offset] = next_m;
      updated_unique_values[v_offset] = next_v;
    }
  }
}

template<typename T, typename K, typename IDX>
__global__ void InitValueKernel(uint64_t seed, one::CUDAGeneratorState* cuda_gen_state,
                                uint64_t inc_offset, const int64_t line_size,
                                const int64_t embedding_size, InitParam param,
                                const IDX* column_ids, const uint32_t* num_missing_keys,
                                const uint32_t* missing_indices, T* values) {
  int32_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, global_thread_id, cuda_gen_state->dev_offset, &state);
  for (int row = blockIdx.x; row < *num_missing_keys; row += gridDim.x) {
    const uint32_t index = missing_indices[row];
    const int32_t slot_idx = column_ids[index];
    assert(slot_idx < param.num_columns);
    for (int col = threadIdx.x; col < line_size; col += blockDim.x) {
      const int64_t offset = index * line_size + col;
      T value = 0;
      if (col < embedding_size) {
        if (param.initializer[slot_idx].type == embedding::InitializerType::kUniform) {
          const float low = param.initializer[slot_idx].uniform_param.low;
          const float high = param.initializer[slot_idx].uniform_param.high;
          T rand_num = curand_uniform(&state);
          value = rand_num * (high - low) + low;
        } else if (param.initializer[slot_idx].type == embedding::InitializerType::kNormal) {
          const float mean = param.initializer[slot_idx].normal_param.mean;
          const float std = param.initializer[slot_idx].normal_param.std;
          value = (curand_normal(&state) + mean) / std;
        } else {
          __trap();
        }
      }
      values[offset] = value;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int32_t new_counter = cuda::atomic::Add(&cuda_gen_state->dev_counter, 1) + 1;
    if (new_counter == gridDim.x) {
      cuda_gen_state->dev_counter = 0;           // reset counter to zero
      cuda_gen_state->dev_offset += inc_offset;  // maintain the state of generator's dev_offset
    }
  }
}

template<typename T, typename K>
class PrefetchTmpBufferManager final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PrefetchTmpBufferManager);
  PrefetchTmpBufferManager(void* ptr, const int64_t num_keys, const int64_t value_size)
      : ptr_(ptr) {
    const size_t num_store_missing_bytes = GetCudaAlignedSize(1 * sizeof(uint32_t));
    const size_t store_missing_indices_bytes = GetCudaAlignedSize(num_keys * sizeof(uint32_t));
    const size_t store_values_bytes = GetCudaAlignedSize(num_keys * value_size * sizeof(T));

    num_store_missing_offset_ = 0;
    store_missing_indices_offset_ = num_store_missing_offset_ + num_store_missing_bytes;
    store_values_offset_ = store_missing_indices_offset_ + store_missing_indices_bytes;
    total_buffer_size_ = num_store_missing_bytes + store_missing_indices_bytes + store_values_bytes;
  }
  ~PrefetchTmpBufferManager() = default;

  size_t TotalBufferSize() const { return total_buffer_size_; }

  uint32_t* NumStoreMissingPtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(ptr_) + num_store_missing_offset_);
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

 private:
  size_t num_store_missing_offset_;
  size_t store_missing_indices_offset_;
  size_t store_values_offset_;
  size_t total_buffer_size_;
  void* ptr_;
};

class EmbeddingPrefetchKernelState final : public user_op::OpKernelState {
 public:
  explicit EmbeddingPrefetchKernelState(user_op::KernelInitContext* ctx)
      : generator_(CHECK_JUST(one::MakeGenerator(DeviceType::kCUDA))) {
    OF_CUDA_CHECK(cudaMallocHost(&host_num_keys_, 1 * sizeof(int32_t)));  // TODO: int32_t->IDX
    embedding::EmbeddingOptions options(ctx->Attr<std::string>("embedding_options"));
    key_value_store_ = Global<EmbeddingMgr>::Get()->GetKeyValueStore(
        options.Name(), ctx->parallel_ctx().parallel_id());
    uint32_t max_query_length =
        ctx->TensorDesc4ArgNameAndIndex("unique_ids", 0)->shape().elem_cnt();
    key_value_store_->ReserveQueryLength(max_query_length);

    const std::vector<embedding::EmbeddingColumn>& columns = options.Columns();
    init_param_.reset(new InitParam);
    CHECK_LE(columns.size(), kMaxColumns);
    init_param_->num_columns = columns.size();
    for (int32_t i = 0; i < columns.size(); ++i) {
      init_param_->initializer[i] = columns.at(i).initializer;
    }
  }
  ~EmbeddingPrefetchKernelState() { OF_CUDA_CHECK(cudaFreeHost(host_num_keys_)); }

  void* HostNumKeys() { return host_num_keys_; }

  embedding::KeyValueStore* KeyValueStore() { return key_value_store_; }

  one::Generator* generator() { return generator_.get(); }

  InitParam* InitParams() { return init_param_.get(); }

 private:
  void* host_num_keys_;
  std::shared_ptr<one::Generator> generator_;
  embedding::KeyValueStore* key_value_store_;
  std::unique_ptr<InitParam> init_param_;
};

class EmbeddingKernelState final : public user_op::OpKernelState {
 public:
  explicit EmbeddingKernelState(user_op::KernelInitContext* ctx) {
    OF_CUDA_CHECK(cudaMallocHost(&host_num_keys_, 1 * sizeof(int32_t)));  // TODO: int32_t->IDX
    key_value_store_ = Global<EmbeddingMgr>::Get()->GetKeyValueStore(
        ctx->Attr<std::string>("embedding_name"), ctx->parallel_ctx().parallel_id());
    uint32_t max_query_length =
        ctx->TensorDesc4ArgNameAndIndex("unique_ids", 0)->shape().elem_cnt();
    key_value_store_->ReserveQueryLength(max_query_length);
  }
  ~EmbeddingKernelState() { OF_CUDA_CHECK(cudaFreeHost(host_num_keys_)); }

  void* HostNumKeys() { return host_num_keys_; }
  embedding::KeyValueStore* KeyValueStore() { return key_value_store_; }

 private:
  void* host_num_keys_;
  embedding::KeyValueStore* key_value_store_;
};

}  // namespace

template<typename T, typename K, typename IDX>
class EmbeddingPrefetchKernel final : public user_op::OpKernel {
 public:
  EmbeddingPrefetchKernel() = default;
  ~EmbeddingPrefetchKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<EmbeddingPrefetchKernelState>(ctx);
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    auto* kernel_state = dynamic_cast<EmbeddingPrefetchKernelState*>(state);
    CHECK(kernel_state != nullptr);
    const auto& generator = kernel_state->generator();
    CHECK_NOTNULL(generator);
    std::shared_ptr<one::CUDAGeneratorImpl> cuda_generator =
        CHECK_JUST(generator->Get<one::CUDAGeneratorImpl>());
    uint64_t seed = cuda_generator->current_seed();
    one::CUDAGeneratorState* cuda_gen_state = cuda_generator->cuda_gen_state();
    embedding::KeyValueStore* store = kernel_state->KeyValueStore();
    InitParam* init_param = kernel_state->InitParams();

    const user_op::Tensor* num_unique_ids = ctx->Tensor4ArgNameAndIndex("num_unique_ids", 0);
    const user_op::Tensor* unique_ids = ctx->Tensor4ArgNameAndIndex("unique_ids", 0);
    const user_op::Tensor* column_ids = ctx->Tensor4ArgNameAndIndex("column_ids", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const int64_t embedding_size = ctx->Attr<int64_t>("embedding_size");
    const int64_t line_size = ctx->Attr<int64_t>("line_size");
    PrefetchTmpBufferManager<T, K> buffer_manager(tmp_buffer->mut_dptr(),
                                                  unique_ids->shape().elem_cnt(), line_size);
    uint32_t* host_num_keys = reinterpret_cast<uint32_t*>(kernel_state->HostNumKeys());
    OF_CUDA_CHECK(cudaMemcpyAsync(host_num_keys, num_unique_ids->dptr(), sizeof(IDX),
                                  cudaMemcpyDefault,
                                  ctx->stream()->As<ep::CudaStream>()->cuda_stream()));
    CHECK_JUST(ctx->stream()->Sync());
    uint32_t num_keys = *host_num_keys;
    store->Get(ctx->stream(), num_keys, unique_ids->dptr(), buffer_manager.StoreValuesPtr(),
               buffer_manager.NumStoreMissingPtr(), buffer_manager.StoreMissingIndicesPtr());

    // init values
    const int64_t grid_size = BlocksNum4ThreadsNum(num_keys);
    uint64_t inc_offset = num_keys / grid_size + 1;
    InitValueKernel<T, K, IDX>
        <<<grid_size, line_size, 0, ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
            seed, cuda_gen_state, inc_offset, line_size, embedding_size, *init_param,
            column_ids->dptr<IDX>(), buffer_manager.NumStoreMissingPtr(),
            buffer_manager.StoreMissingIndicesPtr(), buffer_manager.StoreValuesPtr());
    store->Put(ctx->stream(), num_keys, unique_ids->dptr(), buffer_manager.StoreValuesPtr());
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
        const user_op::TensorDesc& unique_ids = ctx->InputTensorDesc("unique_ids", 0);     \
        PrefetchTmpBufferManager<t_dtype, k_dtype> buffer_manager(                         \
            nullptr, unique_ids.shape().elem_cnt(), ctx->Attr<int64_t>("line_size"));      \
        return buffer_manager.TotalBufferSize();                                           \
      });

REGISTER_CUDA_EMBEDDING_PREFETCH_KERNEL(float, int64_t, int32_t)

template<typename T, typename K>
class LookupTmpBufferManager final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LookupTmpBufferManager);
  LookupTmpBufferManager(void* ptr, const int64_t num_keys, const int64_t tmp_buf_counts)
      : ptr_(ptr) {
    const size_t num_store_missing_bytes = GetCudaAlignedSize(1 * sizeof(uint32_t));
    const size_t store_missing_indices_bytes = GetCudaAlignedSize(num_keys * sizeof(uint32_t));
    const size_t tmp_buf_bytes = GetCudaAlignedSize(tmp_buf_counts * sizeof(T));

    num_store_missing_offset_ = 0;
    store_missing_indices_offset_ = num_store_missing_offset_ + num_store_missing_bytes;
    tmp_buf_offset_ = store_missing_indices_offset_ + store_missing_indices_bytes;
    total_buffer_size_ = num_store_missing_bytes + store_missing_indices_bytes + tmp_buf_bytes;
  }
  ~LookupTmpBufferManager() = default;

  size_t TotalBufferSize() const { return total_buffer_size_; }

  uint32_t* NumStoreMissingPtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(ptr_) + num_store_missing_offset_);
  }
  uint32_t* StoreMissingIndicesPtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(ptr_)
                                       + store_missing_indices_offset_);
  }
  void* TmpBufPtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<void*>(reinterpret_cast<char*>(ptr_) + tmp_buf_offset_);
  }

 private:
  size_t num_store_missing_offset_;
  size_t store_missing_indices_offset_;
  size_t tmp_buf_offset_;
  size_t total_buffer_size_;
  void* ptr_;
};

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
    embedding::KeyValueStore* store = kernel_state->KeyValueStore();
    const user_op::Tensor* num_unique_ids = ctx->Tensor4ArgNameAndIndex("num_unique_ids", 0);
    const user_op::Tensor* unique_ids = ctx->Tensor4ArgNameAndIndex("unique_ids", 0);
    user_op::Tensor* unique_values = ctx->Tensor4ArgNameAndIndex("unique_values", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);

    bool has_copy_nd = false;
    bool has_cast = false;
    int64_t tmp_buf_counts = 0;
    if (ctx->has_output("embeddings", 0)) {
      user_op::Tensor* embeddings = ctx->Tensor4ArgNameAndIndex("embeddings", 0);
      has_copy_nd = (embeddings->shape().elem_cnt() != unique_values->shape().elem_cnt());
      has_cast = (embeddings->data_type() != unique_values->data_type());
      if (has_copy_nd && has_cast) { tmp_buf_counts = embeddings->shape().elem_cnt(); }
    }

    LookupTmpBufferManager<T, K> buffer_manager(tmp_buffer->mut_dptr(),
                                                unique_ids->shape().elem_cnt(), tmp_buf_counts);
    uint32_t* host_num_keys = reinterpret_cast<uint32_t*>(kernel_state->HostNumKeys());

    OF_CUDA_CHECK(cudaMemcpyAsync(host_num_keys, num_unique_ids->dptr(), sizeof(uint32_t),
                                  cudaMemcpyDefault,
                                  ctx->stream()->As<ep::CudaStream>()->cuda_stream()));
    CHECK_JUST(ctx->stream()->Sync());
    const int64_t host_num_unique_ids = *host_num_keys;

    store->Get(ctx->stream(), *host_num_keys, unique_ids->dptr(), unique_values->mut_dptr(),
               buffer_manager.NumStoreMissingPtr(), buffer_manager.StoreMissingIndicesPtr());
    OF_CUDA_CHECK(cudaMemcpyAsync(host_num_keys, buffer_manager.NumStoreMissingPtr(),
                                  sizeof(uint32_t), cudaMemcpyDefault,
                                  ctx->stream()->As<ep::CudaStream>()->cuda_stream()));
    CHECK_JUST(ctx->stream()->Sync());
    CHECK_EQ(*host_num_keys, 0);  // we think keys must be in cache or kv_store.

    if (ctx->has_output("embeddings", 0)) {
      user_op::Tensor* embeddings = ctx->Tensor4ArgNameAndIndex("embeddings", 0);
      CHECK(has_copy_nd || has_cast);
      if (has_copy_nd) {
        void* copy_nd_out = nullptr;
        if (has_cast) {
          copy_nd_out = buffer_manager.TmpBufPtr();
        } else {
          copy_nd_out = embeddings->mut_dptr();
        }
        const int64_t ndims = unique_values->shape().NumAxes();
        CHECK_EQ(ndims, 2);
        CHECK_LE(host_num_unique_ids, embeddings->shape().At(0));
        CHECK_EQ(embeddings->shape().At(0), unique_values->shape().At(0));
        CHECK_NE(embeddings->shape().At(1), unique_values->shape().At(1));
        DimVector src_pos_vec(ndims, 0);
        DimVector dst_pos_vec(ndims, 0);
        DimVector src_shape = {host_num_unique_ids, unique_values->shape().At(1)};
        DimVector dst_shape = {host_num_unique_ids, embeddings->shape().At(1)};
        DimVector extent_shape = {host_num_unique_ids, embeddings->shape().At(1)};

        std::unique_ptr<ep::primitive::CopyNd> copy_nd_primitive =
            ep::primitive::NewPrimitive<ep::primitive::CopyNdFactory>(DeviceType::kCUDA, ndims);
        CHECK(copy_nd_primitive);
        copy_nd_primitive->Launch(ctx->stream(), unique_values->data_type(), ndims, copy_nd_out,
                                  dst_shape.data(), dst_pos_vec.data(), unique_values->dptr(),
                                  src_shape.data(), src_pos_vec.data(), extent_shape.data());
      }
      if (has_cast) {
        const void* cast_in = nullptr;
        if (has_copy_nd) {
          cast_in = buffer_manager.TmpBufPtr();
        } else {
          cast_in = unique_values->dptr();
        }
        const int64_t cast_elem_count = host_num_unique_ids * embeddings->shape().At(1);
        std::unique_ptr<ep::primitive::Cast> cast_primitive =
            ep::primitive::NewPrimitive<ep::primitive::CastFactory>(
                DeviceType::kCUDA, unique_values->data_type(), embeddings->data_type());
        cast_primitive->Launch(ctx->stream(), cast_in, embeddings->mut_dptr(), cast_elem_count);
      }
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
          && (user_op::HobDataType("unique_values", 0) == GetDataType<t_dtype>::value))   \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                 \
        const user_op::TensorDesc& unique_ids = ctx->InputTensorDesc("unique_ids", 0);    \
        int64_t tmp_buf_counts = 0;                                                       \
        if (ctx->has_output("embeddings", 0)) {                                           \
          user_op::TensorDesc* unique_values = ctx->OutputTensorDesc("unique_values", 0); \
          user_op::TensorDesc* embeddings = ctx->OutputTensorDesc("embeddings", 0);       \
          if (unique_values->shape().elem_cnt() != embeddings->shape().elem_cnt()         \
              && unique_values->data_type() != embeddings->data_type()) {                 \
            tmp_buf_counts = embeddings->shape().elem_cnt();                              \
          }                                                                               \
        }                                                                                 \
        LookupTmpBufferManager<t_dtype, k_dtype> buffer_manager(                          \
            nullptr, unique_ids.shape().elem_cnt(), tmp_buf_counts);                      \
        return buffer_manager.TotalBufferSize();                                          \
      });

REGISTER_CUDA_EMBEDDING_LOOKUP_KERNEL(float, int64_t, int32_t)

template<typename T, typename G, typename IDX>
class SgdEmbeddingUpdateKernel final : public user_op::OpKernel {
 public:
  SgdEmbeddingUpdateKernel() = default;
  ~SgdEmbeddingUpdateKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* num_unique_ids = ctx->Tensor4ArgNameAndIndex("num_unique_ids", 0);
    const user_op::Tensor* unique_embeddings = ctx->Tensor4ArgNameAndIndex("unique_embeddings", 0);
    const user_op::Tensor* embedding_diff = ctx->Tensor4ArgNameAndIndex("embedding_diff", 0);
    user_op::Tensor* updated_unique_embeddings =
        ctx->Tensor4ArgNameAndIndex("updated_unique_embeddings", 0);
    const int64_t embedding_size = ctx->Attr<int64_t>("embedding_size");
    const auto scale = ctx->Attr<double>("scale");

    const user_op::Tensor* learning_rate = ctx->Tensor4ArgNameAndIndex("learning_rate", 0);
    const float* learning_rate_ptr = learning_rate->dptr<float>();
    const T* scale_by_ptr = nullptr;
    if (ctx->has_input("scale_by_tensor", 0)) {
      const user_op::Tensor* scale_by_tensor = ctx->Tensor4ArgNameAndIndex("scale_by_tensor", 0);
      CHECK_EQ(scale_by_tensor->data_type(), unique_embeddings->data_type());
      CHECK_EQ(scale_by_tensor->shape().elem_cnt(), 1);
      scale_by_ptr = scale_by_tensor->dptr<T>();
    }
    const int64_t* skip_if_ptr = nullptr;
    if (ctx->has_input("skip_if", 0)) {
      const user_op::Tensor* skip_if = ctx->Tensor4ArgNameAndIndex("skip_if", 0);
      CHECK_EQ(skip_if->shape().elem_cnt(), 1);
      skip_if_ptr = skip_if->dptr<int64_t>();
    }
    // update kernel
    SGDUpdateKernel<T, G, IDX>
        <<<BlocksNum4ThreadsNum(embedding_diff->shape().elem_cnt()), kCudaThreadsNumPerBlock, 0,
           ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
            embedding_size, scale, num_unique_ids->dptr<IDX>(), learning_rate_ptr, scale_by_ptr,
            skip_if_ptr, embedding_diff->dptr<G>(), unique_embeddings->dptr<T>(),
            updated_unique_embeddings->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_SGD_EMBEDDING_UPDATE_KERNEL(t_dtype, g_type, idx_dtype)             \
  REGISTER_USER_KERNEL("sgd_embedding_update")                                            \
      .SetCreateFn<SgdEmbeddingUpdateKernel<t_dtype, g_type, idx_dtype>>()                \
      .SetIsMatchedHob(                                                                   \
          (user_op::HobDeviceType() == DeviceType::kCUDA)                                 \
          && (user_op::HobDataType("num_unique_ids", 0) == GetDataType<idx_dtype>::value) \
          && (user_op::HobDataType("embedding_diff", 0) == GetDataType<g_type>::value)    \
          && (user_op::HobDataType("unique_embeddings", 0) == GetDataType<t_dtype>::value));

REGISTER_CUDA_SGD_EMBEDDING_UPDATE_KERNEL(float, half, int32_t)
REGISTER_CUDA_SGD_EMBEDDING_UPDATE_KERNEL(float, float, int32_t)

template<typename T, typename G, typename IDX>
class MomentumEmbeddingUpdateKernel final : public user_op::OpKernel {
 public:
  MomentumEmbeddingUpdateKernel() = default;
  ~MomentumEmbeddingUpdateKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* num_unique_ids = ctx->Tensor4ArgNameAndIndex("num_unique_ids", 0);
    const user_op::Tensor* unique_embeddings = ctx->Tensor4ArgNameAndIndex("unique_embeddings", 0);
    const user_op::Tensor* embedding_diff = ctx->Tensor4ArgNameAndIndex("embedding_diff", 0);
    user_op::Tensor* updated_unique_embeddings =
        ctx->Tensor4ArgNameAndIndex("updated_unique_embeddings", 0);
    const int64_t num_axes = unique_embeddings->shape().NumAxes();
    const int64_t line_size = unique_embeddings->shape().At(num_axes - 1);
    const int64_t num_keys = unique_embeddings->shape().elem_cnt() / line_size;
    const int64_t embedding_size = ctx->Attr<int64_t>("embedding_size");
    CHECK_EQ(line_size, embedding_size * 2);
    const auto beta = ctx->Attr<float>("beta");
    const auto scale = ctx->Attr<double>("scale");
    const T* scale_by_ptr = nullptr;
    if (ctx->has_input("scale_by_tensor", 0)) {
      const user_op::Tensor* scale_by_tensor = ctx->Tensor4ArgNameAndIndex("scale_by_tensor", 0);
      CHECK_EQ(scale_by_tensor->data_type(), unique_embeddings->data_type());
      CHECK_EQ(scale_by_tensor->shape().elem_cnt(), 1);
      scale_by_ptr = scale_by_tensor->dptr<T>();
    }
    const user_op::Tensor* learning_rate = ctx->Tensor4ArgNameAndIndex("learning_rate", 0);
    const float* learning_rate_ptr = learning_rate->dptr<float>();
    const int64_t* skip_if_ptr = nullptr;
    if (ctx->has_input("skip_if", 0)) {
      const user_op::Tensor* skip_if = ctx->Tensor4ArgNameAndIndex("skip_if", 0);
      CHECK_EQ(skip_if->shape().elem_cnt(), 1);
      skip_if_ptr = skip_if->dptr<int64_t>();
    }
    // update kernel
    MomentumUpdateKernel<T, G, IDX>
        <<<num_keys, embedding_size, 0, ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
            line_size, embedding_size, scale, beta, num_unique_ids->dptr<IDX>(), learning_rate_ptr,
            scale_by_ptr, skip_if_ptr, embedding_diff->dptr<G>(), unique_embeddings->dptr<T>(),
            updated_unique_embeddings->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_MOMENTUM_EMBEDDING_UPDATE_KERNEL(t_dtype, g_type, idx_dtype)        \
  REGISTER_USER_KERNEL("momentum_embedding_update")                                       \
      .SetCreateFn<MomentumEmbeddingUpdateKernel<t_dtype, g_type, idx_dtype>>()           \
      .SetIsMatchedHob(                                                                   \
          (user_op::HobDeviceType() == DeviceType::kCUDA)                                 \
          && (user_op::HobDataType("num_unique_ids", 0) == GetDataType<idx_dtype>::value) \
          && (user_op::HobDataType("embedding_diff", 0) == GetDataType<g_type>::value)    \
          && (user_op::HobDataType("unique_embeddings", 0) == GetDataType<t_dtype>::value));

REGISTER_CUDA_MOMENTUM_EMBEDDING_UPDATE_KERNEL(float, half, int32_t)
REGISTER_CUDA_MOMENTUM_EMBEDDING_UPDATE_KERNEL(float, float, int32_t)

template<typename T, typename G, typename IDX>
class AdamEmbeddingUpdateKernel final : public user_op::OpKernel {
 public:
  AdamEmbeddingUpdateKernel() = default;
  ~AdamEmbeddingUpdateKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* num_unique_ids = ctx->Tensor4ArgNameAndIndex("num_unique_ids", 0);
    const user_op::Tensor* unique_embeddings = ctx->Tensor4ArgNameAndIndex("unique_embeddings", 0);
    const user_op::Tensor* embedding_diff = ctx->Tensor4ArgNameAndIndex("embedding_diff", 0);
    user_op::Tensor* updated_unique_embeddings =
        ctx->Tensor4ArgNameAndIndex("updated_unique_embeddings", 0);
    const int64_t num_axes = unique_embeddings->shape().NumAxes();
    const int64_t line_size = unique_embeddings->shape().At(num_axes - 1);
    const int64_t num_keys = unique_embeddings->shape().elem_cnt() / line_size;
    const int64_t embedding_size = ctx->Attr<int64_t>("embedding_size");
    CHECK_EQ(line_size, embedding_size * 3);

    const auto beta1 = ctx->Attr<float>("beta1");
    const auto beta2 = ctx->Attr<float>("beta2");
    const auto epsilon = ctx->Attr<float>("epsilon");
    const bool do_bias_correction = ctx->Attr<bool>("do_bias_correction");
    const auto scale = ctx->Attr<double>("scale");
    const T* scale_by_ptr = nullptr;
    if (ctx->has_input("scale_by_tensor", 0)) {
      const user_op::Tensor* scale_by_tensor = ctx->Tensor4ArgNameAndIndex("scale_by_tensor", 0);
      CHECK_EQ(scale_by_tensor->data_type(), unique_embeddings->data_type());
      CHECK_EQ(scale_by_tensor->shape().elem_cnt(), 1);
      scale_by_ptr = scale_by_tensor->dptr<T>();
    }
    const user_op::Tensor* learning_rate = ctx->Tensor4ArgNameAndIndex("learning_rate", 0);
    const float* learning_rate_ptr = learning_rate->dptr<float>();
    const int64_t* skip_if_ptr = nullptr;
    if (ctx->has_input("skip_if", 0)) {
      const user_op::Tensor* skip_if = ctx->Tensor4ArgNameAndIndex("skip_if", 0);
      CHECK_EQ(skip_if->shape().elem_cnt(), 1);
      skip_if_ptr = skip_if->dptr<int64_t>();
    }
    const float* bias_correction1_ptr = nullptr;
    if (ctx->has_input("bias_correction1", 0)) {
      bias_correction1_ptr = ctx->Tensor4ArgNameAndIndex("bias_correction1", 0)->dptr<float>();
    }
    const float* bias_correction2_ptr = nullptr;
    if (ctx->has_input("bias_correction2", 0)) {
      bias_correction2_ptr = ctx->Tensor4ArgNameAndIndex("bias_correction2", 0)->dptr<float>();
    }
    // update kernel
    AdamUpdateKernel<T, G, IDX>
        <<<num_keys, embedding_size, 0, ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
            line_size, embedding_size, static_cast<T>(scale), beta1, beta2, epsilon,
            bias_correction1_ptr, bias_correction2_ptr, num_unique_ids->dptr<IDX>(),
            learning_rate_ptr, scale_by_ptr, skip_if_ptr, embedding_diff->dptr<G>(),
            unique_embeddings->dptr<T>(), updated_unique_embeddings->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_ADAM_EMBEDDING_UPDATE_KERNEL(t_dtype, g_type, idx_dtype)            \
  REGISTER_USER_KERNEL("adam_embedding_update")                                           \
      .SetCreateFn<AdamEmbeddingUpdateKernel<t_dtype, g_type, idx_dtype>>()               \
      .SetIsMatchedHob(                                                                   \
          (user_op::HobDeviceType() == DeviceType::kCUDA)                                 \
          && (user_op::HobDataType("num_unique_ids", 0) == GetDataType<idx_dtype>::value) \
          && (user_op::HobDataType("embedding_diff", 0) == GetDataType<g_type>::value)    \
          && (user_op::HobDataType("unique_embeddings", 0) == GetDataType<t_dtype>::value));

REGISTER_CUDA_ADAM_EMBEDDING_UPDATE_KERNEL(float, half, int32_t)
REGISTER_CUDA_ADAM_EMBEDDING_UPDATE_KERNEL(float, float, int32_t)

template<typename IDX>
class EmbeddingPutKernel final : public user_op::OpKernel {
 public:
  EmbeddingPutKernel() = default;
  ~EmbeddingPutKernel() = default;

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
    embedding::KeyValueStore* store = kernel_state->KeyValueStore();
    const user_op::Tensor* num_unique_ids = ctx->Tensor4ArgNameAndIndex("num_unique_ids", 0);
    const user_op::Tensor* unique_ids = ctx->Tensor4ArgNameAndIndex("unique_ids", 0);
    const user_op::Tensor* unique_embeddings = ctx->Tensor4ArgNameAndIndex("unique_embeddings", 0);

    IDX* host_num_keys = reinterpret_cast<IDX*>(kernel_state->HostNumKeys());
    OF_CUDA_CHECK(cudaMemcpyAsync(host_num_keys, num_unique_ids->dptr(), sizeof(IDX),
                                  cudaMemcpyDefault,
                                  ctx->stream()->As<ep::CudaStream>()->cuda_stream()));
    CHECK_JUST(ctx->stream()->Sync());

    store->Put(ctx->stream(), *host_num_keys, unique_ids->dptr(), unique_embeddings->dptr());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_EMBEDDING_PUT_KERNEL(idx_dtype)     \
  REGISTER_USER_KERNEL("embedding_put")                   \
      .SetCreateFn<EmbeddingPutKernel<idx_dtype>>()       \
      .SetIsMatchedHob(                                   \
          (user_op::HobDeviceType() == DeviceType::kCUDA) \
          && (user_op::HobDataType("num_unique_ids", 0) == GetDataType<idx_dtype>::value));

REGISTER_CUDA_EMBEDDING_PUT_KERNEL(int32_t)

}  // namespace oneflow
