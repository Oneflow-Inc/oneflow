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
#include "oneflow/core/embedding/embedding_manager.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/user/kernels/random_mask_generator.h"
#include "oneflow/core/framework/random_generator_impl.h"
#include "oneflow/core/cuda/atomic.cuh"
#include "oneflow/core/ep/include/primitive/copy_nd.h"
#include "oneflow/core/ep/include/primitive/cast.h"
#include "oneflow/core/ep/include/device.h"

namespace oneflow {

namespace {

enum class InitializerType { kUniform, kNormal, kConstant };

struct EmbeddingInitializer {
  InitializerType type;
  union {
    struct {
      float low;
      float high;
    } uniform_param;
    struct {
      float mean;
      float std;
    } normal_param;
    struct {
      float value;
    } constant_param;
  };

  bool operator==(const EmbeddingInitializer& rhs) const {
    if (this->type != rhs.type) { return false; }
    if (rhs.type == InitializerType::kUniform) {
      return (this->uniform_param.low == rhs.uniform_param.low)
             && (this->uniform_param.high == rhs.uniform_param.high);
    } else if (rhs.type == InitializerType::kNormal) {
      return (this->normal_param.mean == rhs.normal_param.mean)
             && (this->normal_param.std == rhs.normal_param.std);
    } else if (rhs.type == InitializerType::kConstant) {
      return this->constant_param.value == rhs.constant_param.value;
    } else {
      UNIMPLEMENTED();
      return false;
    }
  }
};

void ParseInitializerFromJson(const nlohmann::json& initializer,
                              EmbeddingInitializer* embedding_initializer) {
  CHECK(initializer.contains("type"));
  CHECK(initializer["type"].is_string());
  std::string type = initializer["type"].get<std::string>();
  if (type == "uniform") {
    embedding_initializer->type = InitializerType::kUniform;
    CHECK(initializer.contains("low"));
    CHECK(initializer.contains("high"));
    CHECK(initializer["low"].is_number());
    CHECK(initializer["high"].is_number());
    embedding_initializer->uniform_param.low = initializer["low"];
    embedding_initializer->uniform_param.high = initializer["high"];
  } else if (type == "normal") {
    CHECK(initializer.contains("mean"));
    CHECK(initializer.contains("std"));
    CHECK(initializer["mean"].is_number());
    CHECK(initializer["std"].is_number());
    embedding_initializer->type = InitializerType::kNormal;
    embedding_initializer->normal_param.mean = initializer["mean"];
    embedding_initializer->normal_param.std = initializer["std"];
  } else if (type == "constant") {
    CHECK(initializer.contains("value"));
    CHECK(initializer["value"].is_number());
    embedding_initializer->type = InitializerType::kConstant;
    embedding_initializer->constant_param.value = initializer["value"];
  } else {
    UNIMPLEMENTED() << "Unsupported initializer type";
  }
}

int32_t ParseJsonToUniqueInitializerVecAndReturnOffset(
    const nlohmann::json& initializer, std::vector<EmbeddingInitializer>* initializers) {
  EmbeddingInitializer embedding_initializer;
  ParseInitializerFromJson(initializer, &embedding_initializer);
  for (int32_t i = 0; i < initializers->size(); ++i) {
    if (initializers->at(i) == embedding_initializer) { return i; }
  }
  initializers->push_back(embedding_initializer);
  return initializers->size() - 1;
}

void SetInitializerIndex(int32_t row_id, int32_t col_start, int32_t col_end, int64_t line_size,
                         int8_t index, std::vector<int8_t>* initializer_index) {
  int64_t row_offset = row_id * line_size;
  for (int32_t col = col_start; col < col_end; ++col) {
    initializer_index->at(row_offset + col) = index;
  }
}

void ParseAndSetStateInitializerIndex(const std::string& state_initializer,
                                      const int32_t num_tables, const int64_t line_size,
                                      const int64_t embedding_size,
                                      std::vector<EmbeddingInitializer>* initializer_params,
                                      std::vector<int8_t>* initializer_index) {
  if (line_size == embedding_size) { return; }
  CHECK(!state_initializer.empty());
  auto initializers = nlohmann::json::parse(state_initializer);
  CHECK(initializers.is_array());
  const int num_states = line_size / embedding_size - 1;
  CHECK_EQ(num_states, initializers.size());
  for (int32_t i = 0; i < num_states; ++i) {
    int32_t offset =
        ParseJsonToUniqueInitializerVecAndReturnOffset(initializers.at(i), initializer_params);
    int32_t col_start = embedding_size + i * embedding_size;
    int32_t col_end = col_start + embedding_size;
    CHECK_LE(col_end, line_size);
    for (int32_t j = 0; j < num_tables; ++j) {
      SetInitializerIndex(j, col_start, col_end, line_size, offset, initializer_index);
    }
  }
}

void ParseAndSetModelInitializerIndex(const nlohmann::json& tables,
                                      const std::vector<int64_t>& column_dims,
                                      const int32_t num_tables, const int32_t num_columns,
                                      const int64_t line_size, const int64_t embedding_size,
                                      std::vector<EmbeddingInitializer>* initializer_params,
                                      std::vector<int8_t>* initializer_index) {
  for (int32_t i = 0; i < num_tables; ++i) {
    auto table = tables.at(i);
    CHECK(table.contains("columns"));
    auto columns = table["columns"];
    CHECK(columns.is_array());
    CHECK_EQ(num_columns, columns.size()) << "columns size must equal to num embedding dims";
    int32_t col_start = 0;
    for (int k = 0; k < columns.size(); ++k) {
      auto column = columns.at(k);
      CHECK(column.contains("initializer"));
      int32_t offset =
          ParseJsonToUniqueInitializerVecAndReturnOffset(column["initializer"], initializer_params);
      int32_t col_end = col_start + column_dims.at(k);
      SetInitializerIndex(i, col_start, col_end, line_size, offset, initializer_index);
      col_start = col_end;
    }
    CHECK_EQ(col_start, embedding_size);
  }
}

void ParseInitializers(const int64_t line_size, const int64_t embedding_size,
                       const std::string& state_initializer, const std::string& json_serialized,
                       std::vector<EmbeddingInitializer>* initializer_params,
                       std::vector<int8_t>* initializer_index) {
  auto json_object = nlohmann::json::parse(json_serialized);
  CHECK(json_object.contains("column_dims"));
  std::vector<int64_t> column_dims = json_object["column_dims"];
  const int32_t num_columns = column_dims.size();
  CHECK(json_object.contains("tables"));
  auto tables = json_object["tables"];
  CHECK(tables.is_array());
  const int32_t num_tables = tables.size();
  initializer_index->resize(num_tables * line_size);
  ParseAndSetStateInitializerIndex(state_initializer, num_tables, line_size, embedding_size,
                                   initializer_params, initializer_index);
  ParseAndSetModelInitializerIndex(tables, column_dims, num_tables, num_columns, line_size,
                                   embedding_size, initializer_params, initializer_index);
}

template<typename IDX>
class EmbeddingKernelState final : public user_op::OpKernelState {
 public:
  explicit EmbeddingKernelState(user_op::KernelInitContext* ctx)
      : device_index_(-1), generator_(CHECK_JUST(one::MakeGenerator(DeviceType::kCUDA))) {
    OF_CUDA_CHECK(cudaGetDevice(&device_index_));
    OF_CUDA_CHECK(cudaMallocHost(&host_num_keys_, sizeof(IDX)));
    const std::string& embedding_name = ctx->Attr<std::string>("embedding_name");
    const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
    key_value_store_ = Singleton<embedding::EmbeddingManager>::Get()->GetKeyValueStore(
        embedding_name, parallel_id);
    uint32_t max_query_length =
        ctx->TensorDesc4ArgNameAndIndex("unique_ids", 0)->shape().elem_cnt();
    key_value_store_->ReserveQueryLength(max_query_length);
    embedding_state_ = Singleton<embedding::EmbeddingManager>::Get()->GetEmbeddingState(
        embedding_name, parallel_id);

    const int64_t embedding_size = ctx->Attr<int64_t>("embedding_size");
    const int64_t line_size = ctx->Attr<int64_t>("line_size");
    const std::string& state_initializer = ctx->Attr<std::string>("state_initializer");

    std::vector<EmbeddingInitializer> initializer_param;
    std::vector<int8_t> initializer_index;
    ParseInitializers(line_size, embedding_size, state_initializer,
                      ctx->Attr<std::string>("embedding_tables"), &initializer_param,
                      &initializer_index);

    const size_t param_size_bytes = initializer_param.size() * sizeof(EmbeddingInitializer);
    OF_CUDA_CHECK(cudaMallocHost(&host_initializer_param_, param_size_bytes));
    std::memcpy(host_initializer_param_, initializer_param.data(), param_size_bytes);
    OF_CUDA_CHECK(cudaMalloc(&device_initializer_param_, param_size_bytes));
    OF_CUDA_CHECK(cudaMemcpyAsync(device_initializer_param_, host_initializer_param_,
                                  param_size_bytes, cudaMemcpyDefault,
                                  ctx->stream()->As<ep::CudaStream>()->cuda_stream()));

    const size_t index_size_bytes = initializer_index.size() * sizeof(int8_t);
    OF_CUDA_CHECK(cudaMallocHost(&host_initializer_index_, index_size_bytes));
    std::memcpy(host_initializer_index_, initializer_index.data(), index_size_bytes);
    OF_CUDA_CHECK(cudaMalloc(&device_initializer_index_, index_size_bytes));
    OF_CUDA_CHECK(cudaMemcpyAsync(device_initializer_index_, host_initializer_index_,
                                  index_size_bytes, cudaMemcpyDefault,
                                  ctx->stream()->As<ep::CudaStream>()->cuda_stream()));
  }
  ~EmbeddingKernelState() override {
    CudaCurrentDeviceGuard guard(device_index_);
    OF_CUDA_CHECK(cudaFreeHost(host_num_keys_));
    OF_CUDA_CHECK(cudaFreeHost(host_initializer_param_));
    OF_CUDA_CHECK(cudaFree(device_initializer_param_));
    OF_CUDA_CHECK(cudaFreeHost(host_initializer_index_));
    OF_CUDA_CHECK(cudaFree(device_initializer_index_));
  }

  void* HostNumKeys() { return host_num_keys_; }

  embedding::KeyValueStore* KeyValueStore() { return key_value_store_; }

  embedding::EmbeddingState* EmbeddingState() { return embedding_state_; }

  one::Generator* generator() { return generator_.get(); }

  const int8_t* InitializerIndex() { return device_initializer_index_; }
  const EmbeddingInitializer* Initializers() { return device_initializer_param_; }

 private:
  int device_index_;
  void* host_num_keys_;
  std::shared_ptr<one::Generator> generator_;
  embedding::KeyValueStore* key_value_store_;
  embedding::EmbeddingState* embedding_state_;

  EmbeddingInitializer* host_initializer_param_;
  EmbeddingInitializer* device_initializer_param_;
  int8_t* host_initializer_index_;
  int8_t* device_initializer_index_;
};

class EmbeddingPutKernelState final : public user_op::OpKernelState {
 public:
  explicit EmbeddingPutKernelState(user_op::KernelInitContext* ctx) {
    const std::string& embedding_name = ctx->Attr<std::string>("embedding_name");
    const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
    key_value_store_ = Singleton<embedding::EmbeddingManager>::Get()->GetKeyValueStore(
        embedding_name, parallel_id);
    uint32_t max_query_length =
        ctx->TensorDesc4ArgNameAndIndex("unique_ids", 0)->shape().elem_cnt();
    key_value_store_->ReserveQueryLength(max_query_length);
    embedding_state_ = Singleton<embedding::EmbeddingManager>::Get()->GetEmbeddingState(
        embedding_name, parallel_id);
  }
  ~EmbeddingPutKernelState() override = default;

  embedding::KeyValueStore* KeyValueStore() { return key_value_store_; }
  embedding::EmbeddingState* EmbeddingState() { return embedding_state_; }

 private:
  embedding::KeyValueStore* key_value_store_;
  embedding::EmbeddingState* embedding_state_;
};

template<typename T, typename U>
__global__ void InitValueKernel(uint64_t seed, one::CUDAGeneratorState* cuda_gen_state,
                                uint64_t inc_offset, const int32_t line_size,
                                const int32_t embedding_size,
                                const EmbeddingInitializer* initializer_param,
                                const int8_t* initializer_index, const U* table_ids,
                                const uint32_t* num_missing_keys, const uint32_t* missing_indices,
                                T* values) {
  int32_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, global_thread_id, cuda_gen_state->dev_offset, &state);
  int64_t n = *num_missing_keys * line_size;
  CUDA_1D_KERNEL_LOOP(i, n) {
    int row = i / line_size;
    int col = i - row * line_size;
    const uint32_t index = missing_indices[row];
    const int64_t offset = index * line_size + col;
    const int32_t table_idx = table_ids[index];
    const int32_t initializer_idx = initializer_index[table_idx * line_size + col];
    EmbeddingInitializer initializer = initializer_param[initializer_idx];
    T value;
    if (initializer.type == InitializerType::kUniform) {
      const float low = initializer.uniform_param.low;
      const float high = initializer.uniform_param.high;
      value = curand_uniform(&state) * (high - low) + low;
    } else if (initializer.type == InitializerType::kNormal) {
      const float mean = initializer.normal_param.mean;
      const float std = initializer.normal_param.std;
      value = curand_normal(&state) * std + mean;
    } else if (initializer.type == InitializerType::kConstant) {
      value = initializer.constant_param.value;
    } else {
      __trap();
    }
    values[offset] = value;
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

template<typename T, typename U, typename IDX>
void LookupAndInitMissing(ep::Stream* stream, EmbeddingKernelState<IDX>* kernel_state,
                          uint32_t num_unique, const int64_t embedding_size,
                          const int64_t line_size, const bool is_prefetch, const void* unique_ids,
                          const void* table_ids, void* num_missing_ptr, void* missing_indices,
                          void* store_values) {
  const auto& generator = kernel_state->generator();
  CHECK_NOTNULL(generator);
  std::shared_ptr<one::CUDAGeneratorImpl> cuda_generator =
      CHECK_JUST(generator->template Get<one::CUDAGeneratorImpl>(stream->device()->device_index()));
  uint64_t seed = cuda_generator->current_seed();
  one::CUDAGeneratorState* cuda_gen_state = cuda_generator->cuda_gen_state();
  embedding::KeyValueStore* store = kernel_state->KeyValueStore();
  const EmbeddingInitializer* initializer_param = kernel_state->Initializers();
  const int8_t* initializer_index = kernel_state->InitializerIndex();
  store->Get(stream, num_unique, unique_ids, store_values,
             reinterpret_cast<uint32_t*>(num_missing_ptr),
             reinterpret_cast<uint32_t*>(missing_indices));
  void* host_num_keys = kernel_state->HostNumKeys();
  CHECK_GE(sizeof(IDX), sizeof(uint32_t));  // host_num_keys's buffer size is sizeof(IDX)
  OF_CUDA_CHECK(cudaMemcpyAsync(host_num_keys, num_missing_ptr, sizeof(uint32_t), cudaMemcpyDefault,
                                stream->As<ep::CudaStream>()->cuda_stream()));
  CHECK_JUST(stream->Sync());
  uint32_t num_missing = *reinterpret_cast<uint32_t*>(host_num_keys);
  // init missing values
  if (num_missing > 0) {
    const int64_t elem_cnt = num_missing * line_size;
    const int64_t num_blocks = BlocksNum4ThreadsNum(elem_cnt);
    const uint64_t inc_offset = std::ceil(elem_cnt / num_blocks / kCudaThreadsNumPerBlock);
    InitValueKernel<T, U>
        <<<num_blocks, kCudaThreadsNumPerBlock, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
            seed, cuda_gen_state, inc_offset, line_size, embedding_size, initializer_param,
            initializer_index, reinterpret_cast<const U*>(table_ids),
            reinterpret_cast<uint32_t*>(num_missing_ptr),
            reinterpret_cast<uint32_t*>(missing_indices), reinterpret_cast<T*>(store_values));
  }
  if (is_prefetch) { store->Put(stream, num_unique, unique_ids, store_values); }
}

template<typename T, size_t pack_size>
struct alignas(sizeof(T) * pack_size) Pack {
  T elem[pack_size];
};

template<typename T, typename U, typename V, int pack_size>
__global__ void FusedInitSliceCast(const int32_t elem_cnt, uint64_t seed,
                                   one::CUDAGeneratorState* cuda_gen_state, uint64_t inc_offset,
                                   const int32_t line_size, const int32_t embedding_size,
                                   const int32_t line_num_pack, const int32_t embedding_num_pack,
                                   const EmbeddingInitializer* initializer_param,
                                   const int8_t* initializer_index, const U* table_ids,
                                   const uint8_t* lookup_mask, Pack<T, pack_size>* values,
                                   Pack<V, pack_size>* embeddings) {
  int32_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, global_thread_id, cuda_gen_state->dev_offset, &state);
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    int row = i / line_num_pack;
    int col = i - row * line_num_pack;
    Pack<T, pack_size> value_i;
    if (!lookup_mask[row]) {
      const int32_t table_idx = table_ids[row];
#pragma unroll
      for (int k = 0; k < pack_size; ++k) {
        const int32_t initializer_idx =
            initializer_index[table_idx * line_size + col * pack_size + k];
        EmbeddingInitializer initializer = initializer_param[initializer_idx];
        T value;
        if (initializer.type == InitializerType::kUniform) {
          const float low = initializer.uniform_param.low;
          const float high = initializer.uniform_param.high;
          value = curand_uniform(&state) * (high - low) + low;
        } else if (initializer.type == InitializerType::kNormal) {
          const float mean = initializer.normal_param.mean;
          const float std = initializer.normal_param.std;
          value = curand_normal(&state) * std + mean;
        } else if (initializer.type == InitializerType::kConstant) {
          value = initializer.constant_param.value;
        } else {
          __trap();
        }
        value_i.elem[k] = value;
      }
      values[i] = value_i;
    } else {
      value_i = values[i];
    }
    if (embeddings != nullptr && col < embedding_num_pack) {
      int64_t embedding_offset = row * embedding_num_pack + col;
      Pack<V, pack_size> embedding_i;
#pragma unroll
      for (int k = 0; k < pack_size; ++k) { embedding_i.elem[k] = static_cast<V>(value_i.elem[k]); }
      embeddings[embedding_offset] = embedding_i;
    }
  }
}

template<typename T, typename U, typename V>
void InitMissingAndSliceCast(cudaStream_t cuda_stream, uint32_t num_unique,
                             const int64_t embedding_size, const int64_t line_size, uint64_t seed,
                             one::CUDAGeneratorState* cuda_gen_state,
                             const EmbeddingInitializer* initializer_param,
                             const int8_t* initializer_index, const void* table_ids,
                             const uint8_t* mask, T* values_ptr, V* embeddings_ptr) {
  int32_t pack_size;
  if (embedding_size % 4 == 0 && line_size % 4 == 0) {
    pack_size = 4;
  } else if (embedding_size % 2 == 0 && line_size % 2 == 0) {
    pack_size = 2;
  } else {
    pack_size = 1;
  }
  int32_t embedding_num_pack = embedding_size / pack_size;
  int32_t line_num_pack = line_size / pack_size;
  int64_t value_elem_cnt = num_unique * line_size;
  int64_t value_elem_num_pack = value_elem_cnt / pack_size;
  const int64_t num_blocks = BlocksNum4ThreadsNum(value_elem_num_pack);
  const uint64_t inc_offset = std::ceil(value_elem_cnt / num_blocks / kCudaThreadsNumPerBlock);
  if (pack_size == 4) {
    FusedInitSliceCast<T, U, V, 4><<<num_blocks, kCudaThreadsNumPerBlock, 0, cuda_stream>>>(
        value_elem_num_pack, seed, cuda_gen_state, inc_offset, line_size, embedding_size,
        line_num_pack, embedding_num_pack, initializer_param, initializer_index,
        reinterpret_cast<const U*>(table_ids), mask, reinterpret_cast<Pack<T, 4>*>(values_ptr),
        reinterpret_cast<Pack<V, 4>*>(embeddings_ptr));
  } else if (pack_size == 2) {
    FusedInitSliceCast<T, U, V, 2><<<num_blocks, kCudaThreadsNumPerBlock, 0, cuda_stream>>>(
        value_elem_num_pack, seed, cuda_gen_state, inc_offset, line_size, embedding_size,
        line_num_pack, embedding_num_pack, initializer_param, initializer_index,
        reinterpret_cast<const U*>(table_ids), mask, reinterpret_cast<Pack<T, 2>*>(values_ptr),
        reinterpret_cast<Pack<V, 2>*>(embeddings_ptr));
  } else {
    FusedInitSliceCast<T, U, V, 1><<<num_blocks, kCudaThreadsNumPerBlock, 0, cuda_stream>>>(
        value_elem_num_pack, seed, cuda_gen_state, inc_offset, line_size, embedding_size,
        line_num_pack, embedding_num_pack, initializer_param, initializer_index,
        reinterpret_cast<const U*>(table_ids), mask, reinterpret_cast<Pack<T, 1>*>(values_ptr),
        reinterpret_cast<Pack<V, 1>*>(embeddings_ptr));
  }
}

template<typename T, typename U, typename IDX>
void LookupAndFusedInitMissingSliceCast(ep::Stream* stream, EmbeddingKernelState<IDX>* kernel_state,
                                        uint32_t num_unique, const int64_t embedding_size,
                                        const int64_t line_size, DataType value_dtype,
                                        DataType embedding_dtype, const void* unique_ids,
                                        const void* table_ids, uint8_t* lookup_mask_ptr,
                                        void* values_ptr, void* embeddings_ptr) {
  const auto& generator = kernel_state->generator();
  CHECK_NOTNULL(generator);
  std::shared_ptr<one::CUDAGeneratorImpl> cuda_generator =
      CHECK_JUST(generator->template Get<one::CUDAGeneratorImpl>(stream->device()->device_index()));
  uint64_t seed = cuda_generator->current_seed();
  one::CUDAGeneratorState* cuda_gen_state = cuda_generator->cuda_gen_state();
  embedding::KeyValueStore* store = kernel_state->KeyValueStore();
  const EmbeddingInitializer* initializer_param = kernel_state->Initializers();
  const int8_t* initializer_index = kernel_state->InitializerIndex();
  cudaStream_t cuda_stream = stream->As<ep::CudaStream>()->cuda_stream();
  store->Get(stream, num_unique, unique_ids, values_ptr, lookup_mask_ptr);
  if (embedding_dtype == value_dtype) {
    InitMissingAndSliceCast<T, U, T>(
        cuda_stream, num_unique, embedding_size, line_size, seed, cuda_gen_state, initializer_param,
        initializer_index, reinterpret_cast<const U*>(table_ids), lookup_mask_ptr,
        reinterpret_cast<T*>(values_ptr), reinterpret_cast<T*>(embeddings_ptr));
  } else if (embedding_dtype == DataType::kFloat16) {
    InitMissingAndSliceCast<T, U, half>(
        cuda_stream, num_unique, embedding_size, line_size, seed, cuda_gen_state, initializer_param,
        initializer_index, reinterpret_cast<const U*>(table_ids), lookup_mask_ptr,
        reinterpret_cast<T*>(values_ptr), reinterpret_cast<half*>(embeddings_ptr));
  } else {
    UNIMPLEMENTED() << "Unimplemented data_type " << embedding_dtype;
  }
}

template<typename T, typename U>
__global__ void Copy2D(int64_t out_elem_cnt, const int32_t in_cols, const int32_t out_cols,
                       const T* in, U* out) {
  CUDA_1D_KERNEL_LOOP(i, out_elem_cnt) {
    const int32_t row = i / out_cols;
    const int32_t col = i - row * out_cols;
    const int64_t in_offset = row * in_cols + col;
    out[i] = static_cast<U>(in[in_offset]);
  }
}

template<typename T>
void CopyValuesToEmbeddings(ep::Stream* stream, int64_t num_unique, const int32_t embedding_size,
                            const int32_t value_size, const DataType value_dtype,
                            const DataType embedding_dtype, const T* values, void* embeddings) {
  bool need_cast = (value_dtype != embedding_dtype);
  bool need_copy_nd = (embedding_size != value_size);
  CHECK(need_cast || need_copy_nd);
  if (need_cast && !need_copy_nd) {
    const int64_t cast_elem_count = num_unique * embedding_size;
    std::unique_ptr<ep::primitive::Cast> cast_primitive =
        ep::primitive::NewPrimitive<ep::primitive::CastFactory>(DeviceType::kCUDA, value_dtype,
                                                                embedding_dtype);
    cast_primitive->Launch(stream, values, embeddings, cast_elem_count);
  } else if (!need_cast && need_copy_nd) {
    const int32_t ndims = 2;
    DimVector src_pos_vec(ndims, 0);
    DimVector dst_pos_vec(ndims, 0);
    DimVector src_shape = {num_unique, value_size};
    DimVector dst_shape = {num_unique, embedding_size};
    DimVector extent_shape = {num_unique, embedding_size};
    std::unique_ptr<ep::primitive::CopyNd> copy_nd_primitive =
        ep::primitive::NewPrimitive<ep::primitive::CopyNdFactory>(DeviceType::kCUDA, ndims);
    CHECK(copy_nd_primitive);
    copy_nd_primitive->Launch(stream, value_dtype, ndims, embeddings, dst_shape.data(),
                              dst_pos_vec.data(), values, src_shape.data(), src_pos_vec.data(),
                              extent_shape.data());
  } else {
    const int64_t embedding_elem_cnt = num_unique * embedding_size;
    if (embedding_dtype == DataType::kFloat16) {
      Copy2D<T, half><<<BlocksNum4ThreadsNum(embedding_elem_cnt), kCudaThreadsNumPerBlock, 0,
                        stream->As<ep::CudaStream>()->cuda_stream()>>>(
          embedding_elem_cnt, value_size, embedding_size, values,
          reinterpret_cast<half*>(embeddings));
    } else {
      UNIMPLEMENTED();
    }
  }
}

template<typename T, bool is_prefetch>
user_op::InferTmpSizeFn GenEmbeddingInferTmpSizeFn() {
  return [](user_op::InferContext* ctx) {
    size_t total_buffer_size = 0;
    if (embedding::UseDynamicMemoryAllocation()) { return total_buffer_size; }
    const user_op::TensorDesc& unique_ids = ctx->InputTensorDesc("unique_ids", 0);
    int64_t num_ids = unique_ids.shape().elem_cnt();
    size_t num_missing_size = GetCudaAlignedSize(sizeof(uint32_t));
    size_t missing_indices_size = GetCudaAlignedSize(num_ids * sizeof(uint32_t));
    size_t value_buffer_size;
    if (is_prefetch) {
      size_t value_byte_size = ctx->Attr<int64_t>("line_size") * sizeof(T);
      value_buffer_size = num_ids * value_byte_size;
    } else {
      value_buffer_size = 0;
    }
    total_buffer_size = num_missing_size + missing_indices_size + value_buffer_size;
    return total_buffer_size;
  };
}

}  // namespace

template<typename T, typename U, typename IDX>
class EmbeddingPrefetchKernel final : public user_op::OpKernel {
 public:
  EmbeddingPrefetchKernel() : current_iter_(0){};
  ~EmbeddingPrefetchKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<EmbeddingKernelState<IDX>>(ctx);
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    auto* kernel_state = dynamic_cast<EmbeddingKernelState<IDX>*>(state);
    CHECK(kernel_state != nullptr);
    embedding::EmbeddingState* embedding_state = kernel_state->EmbeddingState();
    embedding_state->OnEmbeddingPrefetchStart(ctx, current_iter_);
    uint32_t num_unique = embedding_state->GetIdNumUnique(current_iter_);
    const user_op::Tensor* num_unique_ids = ctx->Tensor4ArgNameAndIndex("num_unique_ids", 0);
    const user_op::Tensor* unique_ids = ctx->Tensor4ArgNameAndIndex("unique_ids", 0);
    const user_op::Tensor* table_ids = ctx->Tensor4ArgNameAndIndex("table_ids", 0);
    const int64_t embedding_size = ctx->Attr<int64_t>("embedding_size");
    const int64_t line_size = ctx->Attr<int64_t>("line_size");

    void* num_missing_ptr;
    embedding_state->AllocPrefetchTmpBuffer(ctx, &num_missing_ptr,
                                            GetCudaAlignedSize(sizeof(uint32_t)));
    void* missing_indices_ptr;
    embedding_state->AllocPrefetchTmpBuffer(ctx, &missing_indices_ptr,
                                            GetCudaAlignedSize(num_unique * sizeof(uint32_t)));
    void* values_ptr;
    embedding_state->AllocPrefetchTmpBuffer(ctx, &values_ptr,
                                            GetCudaAlignedSize(num_unique * line_size * sizeof(T)));
    LookupAndInitMissing<T, U, IDX>(ctx->stream(), kernel_state, num_unique, embedding_size,
                                    line_size, true, unique_ids->dptr(), table_ids->dptr(),
                                    num_missing_ptr, missing_indices_ptr, values_ptr);
    embedding_state->FreePrefetchTmpBuffer(ctx, num_missing_ptr);
    embedding_state->FreePrefetchTmpBuffer(ctx, missing_indices_ptr);
    embedding_state->FreePrefetchTmpBuffer(ctx, values_ptr);
    embedding_state->OnEmbeddingPrefetchEnd(ctx, current_iter_);
    current_iter_++;
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  mutable int64_t current_iter_;
};

#define EMBEDDING_DATA_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(float, DataType::kFloat)

#define TABLE_ID_DATA_TYPE_SEQ                      \
  OF_PP_MAKE_TUPLE_SEQ(uint8_t, DataType::kUInt8)   \
  OF_PP_MAKE_TUPLE_SEQ(uint32_t, DataType::kUInt32) \
  OF_PP_MAKE_TUPLE_SEQ(uint64_t, DataType::kUInt64) \
  OF_PP_MAKE_TUPLE_SEQ(int8_t, DataType::kInt8)     \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32)   \
  OF_PP_MAKE_TUPLE_SEQ(int64_t, DataType::kInt64)

#define IDX_DATA_TYPE_SEQ                           \
  OF_PP_MAKE_TUPLE_SEQ(uint32_t, DataType::kUInt32) \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32)

#define REGISTER_CUDA_EMBEDDING_PREFETCH_KERNEL(t_dtype_pair, table_dtype_pair, idx_dtype_pair) \
  REGISTER_USER_KERNEL("embedding_prefetch")                                                    \
      .SetCreateFn<EmbeddingPrefetchKernel<OF_PP_PAIR_FIRST(t_dtype_pair),                      \
                                           OF_PP_PAIR_FIRST(table_dtype_pair),                  \
                                           OF_PP_PAIR_FIRST(idx_dtype_pair)>>()                 \
      .SetIsMatchedHob(                                                                         \
          (user_op::HobDeviceType() == DeviceType::kCUDA)                                       \
          && (user_op::HobDataType("table_ids", 0) == OF_PP_PAIR_SECOND(table_dtype_pair))      \
          && (user_op::HobDataType("num_unique_ids", 0) == OF_PP_PAIR_SECOND(idx_dtype_pair)))  \
      .SetInferTmpSizeFn(GenEmbeddingInferTmpSizeFn<OF_PP_PAIR_FIRST(t_dtype_pair), true>());

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_CUDA_EMBEDDING_PREFETCH_KERNEL, EMBEDDING_DATA_TYPE_SEQ,
                                 TABLE_ID_DATA_TYPE_SEQ, IDX_DATA_TYPE_SEQ)

template<typename T, typename U, typename IDX>
class EmbeddingLookupKernel final : public user_op::OpKernel {
 public:
  EmbeddingLookupKernel() : current_iter_(0){};
  ~EmbeddingLookupKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<EmbeddingKernelState<IDX>>(ctx);
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    auto* kernel_state = dynamic_cast<EmbeddingKernelState<IDX>*>(state);
    CHECK(kernel_state != nullptr);
    embedding::EmbeddingState* embedding_state = kernel_state->EmbeddingState();
    embedding_state->OnEmbeddingLookupStart(ctx, current_iter_);
    const user_op::Tensor* num_unique_ids = ctx->Tensor4ArgNameAndIndex("num_unique_ids", 0);
    const user_op::Tensor* unique_ids = ctx->Tensor4ArgNameAndIndex("unique_ids", 0);
    const user_op::Tensor* table_ids = ctx->Tensor4ArgNameAndIndex("table_ids", 0);
    user_op::Tensor* unique_values = ctx->Tensor4ArgNameAndIndex("unique_values", 0);
    const int64_t embedding_size = ctx->Attr<int64_t>("embedding_size");
    const int64_t line_size = ctx->Attr<int64_t>("line_size");
    const bool has_output_embeddings = ctx->has_output("embeddings", 0);
    uint32_t num_unique = embedding_state->GetIdNumUnique(current_iter_);
    void* values_ptr = embedding_state->LookupUniqueValues(current_iter_);
    if (has_output_embeddings && kernel_state->KeyValueStore()->IsFusionSupported()) {
      void* embeddings_ptr = embedding_state->LookupEmbeddings(current_iter_);
      user_op::Tensor* embeddings = ctx->Tensor4ArgNameAndIndex("embeddings", 0);
      void* lookup_mask_ptr;
      embedding_state->AllocTmpBuffer(ctx, &lookup_mask_ptr,
                                      GetCudaAlignedSize(num_unique * sizeof(uint8_t)));
      LookupAndFusedInitMissingSliceCast<T, U, IDX>(
          ctx->stream(), kernel_state, num_unique, embedding_size, line_size,
          unique_values->data_type(), embeddings->data_type(), unique_ids->dptr(),
          table_ids->dptr(), reinterpret_cast<uint8_t*>(lookup_mask_ptr), values_ptr,
          embeddings_ptr);
      embedding_state->FreeTmpBuffer(ctx, lookup_mask_ptr);
    } else {
      void* num_missing_ptr;
      embedding_state->AllocTmpBuffer(ctx, &num_missing_ptr, GetCudaAlignedSize(sizeof(uint32_t)));
      void* missing_indices_ptr;
      embedding_state->AllocTmpBuffer(ctx, &missing_indices_ptr,
                                      GetCudaAlignedSize(num_unique * sizeof(uint32_t)));
      LookupAndInitMissing<T, U, IDX>(ctx->stream(), kernel_state, num_unique, embedding_size,
                                      line_size, false, unique_ids->dptr(), table_ids->dptr(),
                                      num_missing_ptr, missing_indices_ptr, values_ptr);
      embedding_state->FreeTmpBuffer(ctx, num_missing_ptr);
      embedding_state->FreeTmpBuffer(ctx, missing_indices_ptr);
      if (has_output_embeddings) {
        void* embeddings_ptr = embedding_state->LookupEmbeddings(current_iter_);
        user_op::Tensor* embeddings = ctx->Tensor4ArgNameAndIndex("embeddings", 0);
        CopyValuesToEmbeddings<T>(ctx->stream(), num_unique, embedding_size, line_size,
                                  unique_values->data_type(), embeddings->data_type(),
                                  reinterpret_cast<T*>(values_ptr), embeddings_ptr);
      }
    }
    embedding_state->OnEmbeddingLookupEnd(ctx, current_iter_);
    current_iter_++;
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  mutable int64_t current_iter_;
};

#define REGISTER_CUDA_EMBEDDING_LOOKUP_KERNEL(t_dtype_pair, table_dtype_pair, idx_dtype_pair)  \
  REGISTER_USER_KERNEL("embedding_lookup")                                                     \
      .SetCreateFn<EmbeddingLookupKernel<OF_PP_PAIR_FIRST(t_dtype_pair),                       \
                                         OF_PP_PAIR_FIRST(table_dtype_pair),                   \
                                         OF_PP_PAIR_FIRST(idx_dtype_pair)>>()                  \
      .SetIsMatchedHob(                                                                        \
          (user_op::HobDeviceType() == DeviceType::kCUDA)                                      \
          && (user_op::HobDataType("unique_values", 0) == OF_PP_PAIR_SECOND(t_dtype_pair))     \
          && (user_op::HobDataType("table_ids", 0) == OF_PP_PAIR_SECOND(table_dtype_pair))     \
          && (user_op::HobDataType("num_unique_ids", 0) == OF_PP_PAIR_SECOND(idx_dtype_pair))) \
      .SetInferTmpSizeFn(GenEmbeddingInferTmpSizeFn<OF_PP_PAIR_FIRST(t_dtype_pair), false>());

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_CUDA_EMBEDDING_LOOKUP_KERNEL, EMBEDDING_DATA_TYPE_SEQ,
                                 TABLE_ID_DATA_TYPE_SEQ, IDX_DATA_TYPE_SEQ)

template<typename IDX>
class EmbeddingPutKernel final : public user_op::OpKernel {
 public:
  EmbeddingPutKernel() : current_iter_(0){};
  ~EmbeddingPutKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<EmbeddingPutKernelState>(ctx);
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    auto* kernel_state = dynamic_cast<EmbeddingPutKernelState*>(state);
    CHECK(kernel_state != nullptr);
    embedding::KeyValueStore* store = kernel_state->KeyValueStore();
    embedding::EmbeddingState* embedding_state = kernel_state->EmbeddingState();
    embedding_state->OnEmbeddingPutStart(ctx, current_iter_);
    const user_op::Tensor* num_unique_ids = ctx->Tensor4ArgNameAndIndex("num_unique_ids", 0);
    const user_op::Tensor* unique_ids = ctx->Tensor4ArgNameAndIndex("unique_ids", 0);
    const user_op::Tensor* unique_embeddings = ctx->Tensor4ArgNameAndIndex("unique_embeddings", 0);
    uint32_t num_unique = embedding_state->GetIdNumUnique(current_iter_);
    store->Put(ctx->stream(), num_unique, unique_ids->dptr(),
               embedding_state->EmbeddingPutUniqueEmbeddings(current_iter_));
    embedding_state->OnEmbeddingPutEnd(ctx, current_iter_);
    current_iter_++;
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  mutable int64_t current_iter_;
};

#define REGISTER_CUDA_EMBEDDING_PUT_KERNEL(dtype, typeproto)           \
  REGISTER_USER_KERNEL("embedding_put")                                \
      .SetCreateFn<EmbeddingPutKernel<dtype>>()                        \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("num_unique_ids", 0) == typeproto));

OF_PP_FOR_EACH_TUPLE(REGISTER_CUDA_EMBEDDING_PUT_KERNEL, IDX_DATA_TYPE_SEQ)

template<typename IDX>
class FusedSgdEmbeddingUpdatePutKernel final : public user_op::OpKernel {
 public:
  FusedSgdEmbeddingUpdatePutKernel() : current_iter_(0){};
  ~FusedSgdEmbeddingUpdatePutKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<EmbeddingPutKernelState>(ctx);
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    auto* kernel_state = dynamic_cast<EmbeddingPutKernelState*>(state);
    CHECK(kernel_state != nullptr);
    embedding::KeyValueStore* store = kernel_state->KeyValueStore();
    embedding::EmbeddingState* embedding_state = kernel_state->EmbeddingState();
    embedding_state->OnEmbeddingFusedUpdatePutStart(ctx, current_iter_);
    const user_op::Tensor* unique_ids = ctx->Tensor4ArgNameAndIndex("unique_ids", 0);
    const user_op::Tensor* embedding_grad = ctx->Tensor4ArgNameAndIndex("embedding_grad", 0);
    const user_op::Tensor* learning_rate = ctx->Tensor4ArgNameAndIndex("learning_rate", 0);
    const float* learning_rate_ptr = learning_rate->dptr<float>();
    const auto scale = ctx->Attr<double>("scale");
    uint32_t num_unique = embedding_state->GetIdNumUnique(current_iter_);
    store->FusedHalfUpdatePut(
        ctx->stream(), num_unique, unique_ids->dptr(),
        embedding_state->EmbeddingFusedUpdatePutUniqueEmbeddings(current_iter_),
        embedding_grad->dptr(), learning_rate_ptr, scale);
    embedding_state->OnEmbeddingFusedUpdatePutEnd(ctx, current_iter_);
    current_iter_++;
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  mutable int64_t current_iter_;
};

#define REGISTER_CUDA_FUSED_SGD_EMBEDDING_UPDATE_PUT_KERNEL(dtype, typeproto)                \
  REGISTER_USER_KERNEL("fused_sgd_embedding_update_put")                                     \
      .SetCreateFn<FusedSgdEmbeddingUpdatePutKernel<dtype>>()                                \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                       \
                       && (user_op::HobDataType("num_unique_ids", 0) == typeproto)           \
                       && (user_op::HobDataType("unique_embeddings", 0) == DataType::kFloat) \
                       && (user_op::HobDataType("embedding_grad", 0) == DataType::kFloat16));

OF_PP_FOR_EACH_TUPLE(REGISTER_CUDA_FUSED_SGD_EMBEDDING_UPDATE_PUT_KERNEL, IDX_DATA_TYPE_SEQ)

}  // namespace oneflow
