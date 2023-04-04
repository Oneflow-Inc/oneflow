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
#include "oneflow/core/ep/include/primitive/copy_nd.h"
#include "oneflow/core/ep/include/primitive/cast.h"
#include "oneflow/core/ep/include/device.h"
#include "oneflow/user/kernels/one_embedding_data_shuffle.cuh"
#include <curand.h>
#include <curand_kernel.h>

namespace oneflow {

namespace {

enum class InitializerType { kUniform, kNormal, kConstant, kTruncNormal };

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
    struct {
      float mean;
      float std;
      float a;
      float b;
    } trunc_normal_param;
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
    } else if (rhs.type == InitializerType::kTruncNormal) {
      return (this->trunc_normal_param.mean == rhs.trunc_normal_param.mean)
             && (this->trunc_normal_param.std == rhs.trunc_normal_param.std)
             && (this->trunc_normal_param.a == rhs.trunc_normal_param.a)
             && (this->trunc_normal_param.b == rhs.trunc_normal_param.b);
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
  } else if (type == "trunc_normal") {
    CHECK(initializer.contains("mean"));
    CHECK(initializer.contains("std"));
    CHECK(initializer.contains("a"));
    CHECK(initializer.contains("b"));
    CHECK(initializer["mean"].is_number());
    CHECK(initializer["std"].is_number());
    CHECK(initializer["a"].is_number());
    CHECK(initializer["b"].is_number());
    embedding_initializer->type = InitializerType::kTruncNormal;
    embedding_initializer->trunc_normal_param.mean = initializer["mean"];
    embedding_initializer->trunc_normal_param.std = initializer["std"];
    embedding_initializer->trunc_normal_param.a = initializer["a"];
    embedding_initializer->trunc_normal_param.b = initializer["b"];
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

void ParseAndSetStepInitializerIndex(const int32_t num_tables, const int64_t line_size,
                                     const int64_t embedding_size,
                                     std::vector<EmbeddingInitializer>* initializer_params,
                                     std::vector<int8_t>* initializer_index) {
  if (line_size % embedding_size == 0) { return; }
  nlohmann::json initializer;
  initializer["type"] = "constant";
  initializer["value"] = 0.0;
  int32_t offset = ParseJsonToUniqueInitializerVecAndReturnOffset(initializer, initializer_params);
  int32_t col_start = line_size / embedding_size * embedding_size;
  int32_t col_end = line_size;
  CHECK_LE(col_end, line_size);
  for (int32_t j = 0; j < num_tables; ++j) {
    SetInitializerIndex(j, col_start, col_end, line_size, offset, initializer_index);
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
  ParseAndSetStepInitializerIndex(num_tables, line_size, embedding_size, initializer_params,
                                  initializer_index);
  ParseAndSetStateInitializerIndex(state_initializer, num_tables, line_size, embedding_size,
                                   initializer_params, initializer_index);
  ParseAndSetModelInitializerIndex(tables, column_dims, num_tables, num_columns, line_size,
                                   embedding_size, initializer_params, initializer_index);
}

template<typename IDX>
class EmbeddingKernelState final : public user_op::OpKernelState {
 public:
  explicit EmbeddingKernelState(user_op::KernelInitContext* ctx) : device_index_(-1) {
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

  const int8_t* InitializerIndex() { return device_initializer_index_; }
  const EmbeddingInitializer* Initializers() { return device_initializer_param_; }

 private:
  int device_index_;
  void* host_num_keys_;
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

template<typename T, typename K, typename U>
__global__ void InitValueKernel(uint64_t seed, const int32_t line_size,
                                const int32_t embedding_size,
                                const EmbeddingInitializer* initializer_param,
                                const int8_t* initializer_index, const K* unique_ids,
                                const U* table_ids, const uint32_t* num_missing_keys,
                                const uint32_t* missing_indices, T* values) {
  int64_t n = *num_missing_keys * line_size;
  CUDA_1D_KERNEL_LOOP(i, n) {
    int row = i / line_size;
    int col = i - row * line_size;
    const uint32_t index = missing_indices[row];
    const int64_t offset = index * line_size + col;
    const int32_t table_idx = table_ids[index];
    const K id = unique_ids[index];
    curandStatePhilox4_32_10_t state;
    curand_init(seed, id, col, &state);
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
    } else if (initializer.type == InitializerType::kTruncNormal) {
      const float mean = initializer.trunc_normal_param.mean;
      const float std = initializer.trunc_normal_param.std;
      const float a = initializer.trunc_normal_param.a;
      const float b = initializer.trunc_normal_param.b;
      while (true) {
        value = curand_normal(&state) * std + mean;
        if (value >= a && value <= b) { break; }
        skipahead(line_size, &state);
      }
    } else {
      __trap();
    }
    values[offset] = value;
  }
}

template<typename T, typename K, typename U, typename IDX>
void LookupAndInitMissing(ep::Stream* stream, uint64_t seed, embedding::KeyValueStore* store,
                          const EmbeddingInitializer* initializer_param,
                          const int8_t* initializer_index, void* host_num_keys, uint32_t num_unique,
                          const int64_t embedding_size, const int64_t line_size,
                          const bool put_to_store, const void* unique_ids, const void* table_ids,
                          void* num_missing_ptr, void* missing_indices, void* store_values) {
  store->Get(stream, num_unique, unique_ids, store_values,
             reinterpret_cast<uint32_t*>(num_missing_ptr),
             reinterpret_cast<uint32_t*>(missing_indices));
  CHECK_GE(sizeof(IDX), sizeof(uint32_t));  // host_num_keys's buffer size is sizeof(IDX)
  OF_CUDA_CHECK(cudaMemcpyAsync(host_num_keys, num_missing_ptr, sizeof(uint32_t), cudaMemcpyDefault,
                                stream->As<ep::CudaStream>()->cuda_stream()));
  CHECK_JUST(stream->Sync());
  uint32_t num_missing = *reinterpret_cast<uint32_t*>(host_num_keys);
  // init missing values
  if (num_missing > 0) {
    const int64_t elem_cnt = num_missing * line_size;
    const int64_t num_blocks = BlocksNum4ThreadsNum(elem_cnt);
    InitValueKernel<T, K, U>
        <<<num_blocks, kCudaThreadsNumPerBlock, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
            seed, line_size, embedding_size, initializer_param, initializer_index,
            reinterpret_cast<const K*>(unique_ids), reinterpret_cast<const U*>(table_ids),
            reinterpret_cast<uint32_t*>(num_missing_ptr),
            reinterpret_cast<uint32_t*>(missing_indices), reinterpret_cast<T*>(store_values));
  }
  if (put_to_store) { store->Put(stream, num_unique, unique_ids, store_values); }
}

template<typename T, typename K, typename U, typename IDX>
void LookupAndInitMissing(ep::Stream* stream, EmbeddingKernelState<IDX>* kernel_state,
                          uint64_t seed, uint32_t num_unique, const int64_t embedding_size,
                          const int64_t line_size, const bool put_to_store, const void* unique_ids,
                          const void* table_ids, void* num_missing_ptr, void* missing_indices,
                          void* store_values) {
  embedding::KeyValueStore* store = kernel_state->KeyValueStore();
  const EmbeddingInitializer* initializer_param = kernel_state->Initializers();
  const int8_t* initializer_index = kernel_state->InitializerIndex();
  void* host_num_keys = kernel_state->HostNumKeys();
  LookupAndInitMissing<T, K, U, IDX>(stream, seed, store, initializer_param, initializer_index,
                                     host_num_keys, num_unique, embedding_size, line_size,
                                     put_to_store, unique_ids, table_ids, num_missing_ptr,
                                     missing_indices, store_values);
}

template<typename T, size_t pack_size>
struct alignas(sizeof(T) * pack_size) Pack {
  T elem[pack_size];
};

template<typename T, typename K, typename U, typename V, int pack_size>
__global__ void FusedInitSliceCast(const int32_t elem_cnt, uint64_t seed, const int32_t line_size,
                                   const int32_t embedding_size, const int32_t line_num_pack,
                                   const int32_t embedding_num_pack,
                                   const EmbeddingInitializer* initializer_param,
                                   const int8_t* initializer_index, const K* unique_ids,
                                   const U* table_ids, const uint8_t* lookup_mask,
                                   Pack<T, pack_size>* values, Pack<V, pack_size>* embeddings) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    int row = i / line_num_pack;
    int col = i - row * line_num_pack;
    Pack<T, pack_size> value_i;
    if (!lookup_mask[row]) {
      const int32_t table_idx = table_ids[row];
      const K id = unique_ids[row];
      curandStatePhilox4_32_10_t state;
      curand_init(seed, id, col, &state);
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
        } else if (initializer.type == InitializerType::kTruncNormal) {
          const float mean = initializer.trunc_normal_param.mean;
          const float std = initializer.trunc_normal_param.std;
          const float a = initializer.trunc_normal_param.a;
          const float b = initializer.trunc_normal_param.b;
          while (true) {
            value = curand_normal(&state) * std + mean;
            if (value >= a && value <= b) { break; }
            skipahead(line_size, &state);
          }
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

template<typename T, typename K, typename U, typename V>
void InitMissingAndSliceCast(cudaStream_t cuda_stream, uint32_t num_unique,
                             const int64_t embedding_size, const int64_t line_size, uint64_t seed,
                             const EmbeddingInitializer* initializer_param,
                             const int8_t* initializer_index, const void* unique_ids,
                             const void* table_ids, const uint8_t* mask, T* values_ptr,
                             V* embeddings_ptr) {
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
  if (pack_size == 4) {
    FusedInitSliceCast<T, K, U, V, 4><<<num_blocks, kCudaThreadsNumPerBlock, 0, cuda_stream>>>(
        value_elem_num_pack, seed, line_size, embedding_size, line_num_pack, embedding_num_pack,
        initializer_param, initializer_index, reinterpret_cast<const K*>(unique_ids),
        reinterpret_cast<const U*>(table_ids), mask, reinterpret_cast<Pack<T, 4>*>(values_ptr),
        reinterpret_cast<Pack<V, 4>*>(embeddings_ptr));
  } else if (pack_size == 2) {
    FusedInitSliceCast<T, K, U, V, 2><<<num_blocks, kCudaThreadsNumPerBlock, 0, cuda_stream>>>(
        value_elem_num_pack, seed, line_size, embedding_size, line_num_pack, embedding_num_pack,
        initializer_param, initializer_index, reinterpret_cast<const K*>(unique_ids),
        reinterpret_cast<const U*>(table_ids), mask, reinterpret_cast<Pack<T, 2>*>(values_ptr),
        reinterpret_cast<Pack<V, 2>*>(embeddings_ptr));
  } else {
    FusedInitSliceCast<T, K, U, V, 1><<<num_blocks, kCudaThreadsNumPerBlock, 0, cuda_stream>>>(
        value_elem_num_pack, seed, line_size, embedding_size, line_num_pack, embedding_num_pack,
        initializer_param, initializer_index, reinterpret_cast<const K*>(unique_ids),
        reinterpret_cast<const U*>(table_ids), mask, reinterpret_cast<Pack<T, 1>*>(values_ptr),
        reinterpret_cast<Pack<V, 1>*>(embeddings_ptr));
  }
}

template<typename T, typename K, typename U, typename IDX>
void LookupAndFusedInitMissingSliceCast(ep::Stream* stream, EmbeddingKernelState<IDX>* kernel_state,
                                        uint64_t seed, uint32_t num_unique,
                                        const int64_t embedding_size, const int64_t line_size,
                                        DataType value_dtype, DataType embedding_dtype,
                                        const void* unique_ids, const void* table_ids,
                                        uint8_t* lookup_mask_ptr, void* values_ptr,
                                        void* embeddings_ptr) {
  embedding::KeyValueStore* store = kernel_state->KeyValueStore();
  const EmbeddingInitializer* initializer_param = kernel_state->Initializers();
  const int8_t* initializer_index = kernel_state->InitializerIndex();
  cudaStream_t cuda_stream = stream->As<ep::CudaStream>()->cuda_stream();
  store->Get(stream, num_unique, unique_ids, values_ptr, lookup_mask_ptr);
  if (embedding_dtype == value_dtype) {
    InitMissingAndSliceCast<T, K, U, T>(
        cuda_stream, num_unique, embedding_size, line_size, seed, initializer_param,
        initializer_index, reinterpret_cast<const K*>(unique_ids),
        reinterpret_cast<const U*>(table_ids), lookup_mask_ptr, reinterpret_cast<T*>(values_ptr),
        reinterpret_cast<T*>(embeddings_ptr));
  } else if (embedding_dtype == DataType::kFloat16) {
    InitMissingAndSliceCast<T, K, U, half>(
        cuda_stream, num_unique, embedding_size, line_size, seed, initializer_param,
        initializer_index, reinterpret_cast<const K*>(unique_ids),
        reinterpret_cast<const U*>(table_ids), lookup_mask_ptr, reinterpret_cast<T*>(values_ptr),
        reinterpret_cast<half*>(embeddings_ptr));
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
      value_buffer_size = GetCudaAlignedSize(num_ids * value_byte_size);
    } else {
      value_buffer_size = 0;
    }
    total_buffer_size = num_missing_size + missing_indices_size + value_buffer_size;
    return total_buffer_size;
  };
}

class IdShuffleCopyOutKernelState final : public user_op::OpKernelState {
 public:
  explicit IdShuffleCopyOutKernelState(user_op::KernelInitContext* ctx) {
    const std::string& embedding_name = ctx->Attr<std::string>("embedding_name");
    const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
    embedding_state_ = Singleton<embedding::EmbeddingManager>::Get()->GetEmbeddingState(
        embedding_name, parallel_id);
  }
  ~IdShuffleCopyOutKernelState() override = default;

  embedding::EmbeddingState* EmbeddingState() { return embedding_state_; }

 private:
  embedding::EmbeddingState* embedding_state_;
};

template<typename K, typename U, typename IDX>
struct IdShuffleCopyOutParam {
  uint32_t final_num_unique_ids;
  const K* cur_rank_unique_ids;
  K* out_cur_rank_unique_ids;
  const U* cur_rank_unique_table_ids;
  U* out_cur_rank_unique_table_ids;
  uint32_t cur_rank_num_ids;
  const IDX* cur_rank_inverse_indices;
  IDX* out_cur_rank_inverse_indices;
  uint32_t num_ids;
  const IDX* inverse_unique_partition_indices;
  IDX* out_inverse_unique_partition_indices;
  uint32_t num_unique_matrix_cnt;
  const IDX* num_unique_matrix;
  IDX* out_num_unique_matrix;
  const IDX* cur_rank_num_unique;
  IDX* out_cur_rank_num_unique;
};

template<typename K, typename U, typename IDX>
__global__ void CopyGpu(IdShuffleCopyOutParam<K, U, IDX> param) {
  CUDA_1D_KERNEL_LOOP_T(uint32_t, i, param.final_num_unique_ids) {
    param.out_cur_rank_unique_ids[i] = param.cur_rank_unique_ids[i];
    param.out_cur_rank_unique_table_ids[i] = param.cur_rank_unique_table_ids[i];
  }
  CUDA_1D_KERNEL_LOOP_T(uint32_t, i, param.cur_rank_num_ids) {
    param.out_cur_rank_inverse_indices[i] = param.cur_rank_inverse_indices[i];
  }
  CUDA_1D_KERNEL_LOOP_T(uint32_t, i, param.num_ids) {
    param.out_inverse_unique_partition_indices[i] = param.inverse_unique_partition_indices[i];
  }
  CUDA_1D_KERNEL_LOOP_T(uint32_t, i, param.num_unique_matrix_cnt) {
    param.out_num_unique_matrix[i] = param.num_unique_matrix[i];
  }
  if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
    *param.out_cur_rank_num_unique = *param.cur_rank_num_unique;
  }
}

}  // namespace

template<typename T, typename K, typename U, typename IDX>
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
    std::unique_ptr<embedding::TmpBufferAllocator> allocator =
        embedding_state->NewTmpBufferAllocator(ctx);
    uint32_t num_unique = embedding_state->GetIdNumUnique(current_iter_);
    const user_op::Tensor* num_unique_ids = ctx->Tensor4ArgNameAndIndex("num_unique_ids", 0);
    const user_op::Tensor* unique_ids = ctx->Tensor4ArgNameAndIndex("unique_ids", 0);
    const user_op::Tensor* table_ids = ctx->Tensor4ArgNameAndIndex("table_ids", 0);
    const int64_t embedding_size = ctx->Attr<int64_t>("embedding_size");
    const int64_t line_size = ctx->Attr<int64_t>("line_size");
    const int64_t seed = ctx->Attr<int64_t>("seed");
    void* num_missing_ptr;
    allocator->Allocate(&num_missing_ptr, sizeof(uint32_t));
    void* missing_indices_ptr;
    allocator->Allocate(&missing_indices_ptr, num_unique * sizeof(uint32_t));
    void* values_ptr;
    allocator->Allocate(&values_ptr, num_unique * line_size * sizeof(T));
    LookupAndInitMissing<T, K, U, IDX>(
        ctx->stream(), kernel_state, seed, num_unique, embedding_size, line_size, true,
        unique_ids->dptr(), table_ids->dptr(), num_missing_ptr, missing_indices_ptr, values_ptr);
    allocator->Free(num_missing_ptr);
    allocator->Free(missing_indices_ptr);
    allocator->Free(values_ptr);
    current_iter_++;
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  mutable int64_t current_iter_;
};

#define EMBEDDING_DATA_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(float, DataType::kFloat)

#define ID_DATA_TYPE_SEQ                            \
  OF_PP_MAKE_TUPLE_SEQ(uint32_t, DataType::kUInt32) \
  OF_PP_MAKE_TUPLE_SEQ(uint64_t, DataType::kUInt64) \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32)   \
  OF_PP_MAKE_TUPLE_SEQ(int64_t, DataType::kInt64)

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

#define REGISTER_CUDA_EMBEDDING_PREFETCH_KERNEL(t_dtype_pair, k_dtype_pair, table_dtype_pair,  \
                                                idx_dtype_pair)                                \
  REGISTER_USER_KERNEL("embedding_prefetch")                                                   \
      .SetCreateFn<EmbeddingPrefetchKernel<                                                    \
          OF_PP_PAIR_FIRST(t_dtype_pair), OF_PP_PAIR_FIRST(k_dtype_pair),                      \
          OF_PP_PAIR_FIRST(table_dtype_pair), OF_PP_PAIR_FIRST(idx_dtype_pair)>>()             \
      .SetIsMatchedHob(                                                                        \
          (user_op::HobDeviceType() == DeviceType::kCUDA)                                      \
          && (user_op::HobDataType("unique_ids", 0) == OF_PP_PAIR_SECOND(k_dtype_pair))        \
          && (user_op::HobDataType("table_ids", 0) == OF_PP_PAIR_SECOND(table_dtype_pair))     \
          && (user_op::HobDataType("num_unique_ids", 0) == OF_PP_PAIR_SECOND(idx_dtype_pair))) \
      .SetInferTmpSizeFn(GenEmbeddingInferTmpSizeFn<OF_PP_PAIR_FIRST(t_dtype_pair), true>());

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_CUDA_EMBEDDING_PREFETCH_KERNEL, EMBEDDING_DATA_TYPE_SEQ,
                                 ID_DATA_TYPE_SEQ, TABLE_ID_DATA_TYPE_SEQ, IDX_DATA_TYPE_SEQ)

template<typename T, typename K, typename U, typename IDX>
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
    std::unique_ptr<embedding::TmpBufferAllocator> allocator =
        embedding_state->NewTmpBufferAllocator(ctx);
    embedding_state->OnEmbeddingLookupStart(ctx, current_iter_);
    const user_op::Tensor* num_unique_ids = ctx->Tensor4ArgNameAndIndex("num_unique_ids", 0);
    const user_op::Tensor* unique_ids = ctx->Tensor4ArgNameAndIndex("unique_ids", 0);
    const user_op::Tensor* table_ids = ctx->Tensor4ArgNameAndIndex("table_ids", 0);
    user_op::Tensor* unique_values = ctx->Tensor4ArgNameAndIndex("unique_values", 0);
    const int64_t embedding_size = ctx->Attr<int64_t>("embedding_size");
    const int64_t line_size = ctx->Attr<int64_t>("line_size");
    const bool has_output_embeddings = ctx->has_output("embeddings", 0);
    const int64_t seed = ctx->Attr<int64_t>("seed");
    uint32_t num_unique = embedding_state->GetIdNumUnique(current_iter_);
    void* values_ptr = embedding_state->LookupUniqueValues(current_iter_);
    if (has_output_embeddings && kernel_state->KeyValueStore()->IsFusionSupported()) {
      void* embeddings_ptr = embedding_state->LookupEmbeddings(current_iter_);
      user_op::Tensor* embeddings = ctx->Tensor4ArgNameAndIndex("embeddings", 0);
      void* lookup_mask_ptr;
      allocator->Allocate(&lookup_mask_ptr, num_unique * sizeof(uint8_t));
      LookupAndFusedInitMissingSliceCast<T, K, U, IDX>(
          ctx->stream(), kernel_state, seed, num_unique, embedding_size, line_size,
          unique_values->data_type(), embeddings->data_type(), unique_ids->dptr(),
          table_ids->dptr(), reinterpret_cast<uint8_t*>(lookup_mask_ptr), values_ptr,
          embeddings_ptr);
      allocator->Free(lookup_mask_ptr);
    } else {
      void* num_missing_ptr;
      allocator->Allocate(&num_missing_ptr, sizeof(uint32_t));
      void* missing_indices_ptr;
      allocator->Allocate(&missing_indices_ptr, num_unique * sizeof(uint32_t));
      LookupAndInitMissing<T, K, U, IDX>(
          ctx->stream(), kernel_state, seed, num_unique, embedding_size, line_size, false,
          unique_ids->dptr(), table_ids->dptr(), num_missing_ptr, missing_indices_ptr, values_ptr);
      allocator->Free(num_missing_ptr);
      allocator->Free(missing_indices_ptr);
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

#define REGISTER_CUDA_EMBEDDING_LOOKUP_KERNEL(t_dtype_pair, k_dtype_pair, table_dtype_pair,    \
                                              idx_dtype_pair)                                  \
  REGISTER_USER_KERNEL("embedding_lookup")                                                     \
      .SetCreateFn<EmbeddingLookupKernel<                                                      \
          OF_PP_PAIR_FIRST(t_dtype_pair), OF_PP_PAIR_FIRST(k_dtype_pair),                      \
          OF_PP_PAIR_FIRST(table_dtype_pair), OF_PP_PAIR_FIRST(idx_dtype_pair)>>()             \
      .SetIsMatchedHob(                                                                        \
          (user_op::HobDeviceType() == DeviceType::kCUDA)                                      \
          && (user_op::HobDataType("unique_values", 0) == OF_PP_PAIR_SECOND(t_dtype_pair))     \
          && (user_op::HobDataType("unique_ids", 0) == OF_PP_PAIR_SECOND(k_dtype_pair))        \
          && (user_op::HobDataType("table_ids", 0) == OF_PP_PAIR_SECOND(table_dtype_pair))     \
          && (user_op::HobDataType("num_unique_ids", 0) == OF_PP_PAIR_SECOND(idx_dtype_pair))) \
      .SetInferTmpSizeFn(GenEmbeddingInferTmpSizeFn<OF_PP_PAIR_FIRST(t_dtype_pair), false>());

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_CUDA_EMBEDDING_LOOKUP_KERNEL, EMBEDDING_DATA_TYPE_SEQ,
                                 ID_DATA_TYPE_SEQ, TABLE_ID_DATA_TYPE_SEQ, IDX_DATA_TYPE_SEQ)

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
class OneEmbeddingFusedSgdUpdatePutKernel final : public user_op::OpKernel {
 public:
  OneEmbeddingFusedSgdUpdatePutKernel() : current_iter_(0){};
  ~OneEmbeddingFusedSgdUpdatePutKernel() override = default;

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

#define REGISTER_CUDA_ONE_EMBEDDING_FUSED_SGD_UPDATE_PUT_KERNEL(dtype, typeproto)            \
  REGISTER_USER_KERNEL("one_embedding_fused_sgd_update_put")                                 \
      .SetCreateFn<OneEmbeddingFusedSgdUpdatePutKernel<dtype>>()                             \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                       \
                       && (user_op::HobDataType("num_unique_ids", 0) == typeproto)           \
                       && (user_op::HobDataType("unique_embeddings", 0) == DataType::kFloat) \
                       && (user_op::HobDataType("embedding_grad", 0) == DataType::kFloat16));

OF_PP_FOR_EACH_TUPLE(REGISTER_CUDA_ONE_EMBEDDING_FUSED_SGD_UPDATE_PUT_KERNEL, IDX_DATA_TYPE_SEQ)

template<typename K, typename U, typename IDX>
class IdShuffleCopyOutKernel final : public user_op::OpKernel {
 public:
  IdShuffleCopyOutKernel() : current_iter_(0){};
  ~IdShuffleCopyOutKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<IdShuffleCopyOutKernelState>(ctx);
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    auto* kernel_state = dynamic_cast<IdShuffleCopyOutKernelState*>(state);
    CHECK(kernel_state != nullptr);
    const int64_t parallel_num = ctx->parallel_ctx().parallel_num();
    const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
    embedding::EmbeddingState* embedding_state = kernel_state->EmbeddingState();
    const uint32_t num_unique = embedding_state->GetIdNumUnique(current_iter_);
    const std::vector<uint32_t>& num_unique_matrix_vec =
        embedding_state->GetIdNumUniqueMatrix(current_iter_);
    uint32_t cur_rank_num_ids = 0;
    for (int64_t i = 0; i < parallel_num; ++i) {
      cur_rank_num_ids += num_unique_matrix_vec.at(i * parallel_num + parallel_id);
    }
    IdShuffleCopyOutParam<K, U, IDX> param;
    param.final_num_unique_ids = num_unique;
    param.cur_rank_unique_ids =
        reinterpret_cast<const K*>(ctx->Tensor4ArgNameAndIndex("cur_rank_unique_ids", 0)->dptr());
    param.out_cur_rank_unique_ids =
        reinterpret_cast<K*>(ctx->Tensor4ArgNameAndIndex("out_cur_rank_unique_ids", 0)->mut_dptr());
    param.cur_rank_unique_table_ids = reinterpret_cast<const U*>(
        ctx->Tensor4ArgNameAndIndex("cur_rank_unique_table_ids", 0)->dptr());
    param.out_cur_rank_unique_table_ids = reinterpret_cast<U*>(
        ctx->Tensor4ArgNameAndIndex("out_cur_rank_unique_table_ids", 0)->mut_dptr());
    param.cur_rank_num_ids = cur_rank_num_ids;
    param.cur_rank_inverse_indices = reinterpret_cast<const IDX*>(
        ctx->Tensor4ArgNameAndIndex("cur_rank_inverse_indices", 0)->dptr());
    param.out_cur_rank_inverse_indices = reinterpret_cast<IDX*>(
        ctx->Tensor4ArgNameAndIndex("out_cur_rank_inverse_indices", 0)->mut_dptr());
    param.num_ids =
        ctx->Tensor4ArgNameAndIndex("inverse_unique_partition_indices", 0)->shape_view().elem_cnt();
    param.inverse_unique_partition_indices = reinterpret_cast<const IDX*>(
        ctx->Tensor4ArgNameAndIndex("inverse_unique_partition_indices", 0)->dptr());
    param.out_inverse_unique_partition_indices = reinterpret_cast<IDX*>(
        ctx->Tensor4ArgNameAndIndex("out_inverse_unique_partition_indices", 0)->mut_dptr());
    param.num_unique_matrix_cnt = parallel_num * parallel_num;
    param.num_unique_matrix =
        reinterpret_cast<const IDX*>(ctx->Tensor4ArgNameAndIndex("num_unique_matrix", 0)->dptr());
    param.out_num_unique_matrix =
        reinterpret_cast<IDX*>(ctx->Tensor4ArgNameAndIndex("out_num_unique_matrix", 0)->mut_dptr());
    param.cur_rank_num_unique =
        reinterpret_cast<const IDX*>(ctx->Tensor4ArgNameAndIndex("cur_rank_num_unique", 0)->dptr());
    param.out_cur_rank_num_unique = reinterpret_cast<IDX*>(
        ctx->Tensor4ArgNameAndIndex("out_cur_rank_num_unique", 0)->mut_dptr());

    CopyGpu<K, U, IDX><<<BlocksNum4ThreadsNum(param.num_ids), kCudaThreadsNumPerBlock, 0,
                         ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(param);
    current_iter_++;
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  mutable int64_t current_iter_;
};

#define REGISTER_CUDA_ID_SHUFFLE_COPY_OUT_KERNEL(k_dtype_pair, table_id_dtype_pair,              \
                                                 idx_dtype_pair)                                 \
  REGISTER_USER_KERNEL("id_shuffle_copy_out")                                                    \
      .SetCreateFn<IdShuffleCopyOutKernel<OF_PP_PAIR_FIRST(k_dtype_pair),                        \
                                          OF_PP_PAIR_FIRST(table_id_dtype_pair),                 \
                                          OF_PP_PAIR_FIRST(idx_dtype_pair)>>()                   \
      .SetIsMatchedHob(                                                                          \
          (user_op::HobDeviceType() == DeviceType::kCUDA)                                        \
          && (user_op::HobDataType("cur_rank_unique_ids", 0) == OF_PP_PAIR_SECOND(k_dtype_pair)) \
          && (user_op::HobDataType("cur_rank_unique_table_ids", 0)                               \
              == OF_PP_PAIR_SECOND(table_id_dtype_pair))                                         \
          && (user_op::HobDataType("num_unique_matrix", 0) == OF_PP_PAIR_SECOND(idx_dtype_pair)));

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_CUDA_ID_SHUFFLE_COPY_OUT_KERNEL, ID_DATA_TYPE_SEQ,
                                 TABLE_ID_DATA_TYPE_SEQ, IDX_DATA_TYPE_SEQ)

enum class FusedEmbeddingBufferType {
  // id shuffle
  kNumPartitionedUnique = 0,
  kPartitionedUniqueIds,
  kReceivedIds,
  kTableIds,
  kPartitionedUniqueTableIds,
  kReceivedTableIds,
  kWorkspace,
  kNumUniqueMatrix,
  kInverseUniquePartitionIndices,
  kCurRankNumUnique,
  kCurRankUniqueIds,
  kCurRankUniqueTableIds,
  kCurRankInverseIndices,
  // embedding lookup
  kNumMissing,
  kMissingIndices,
  kCurRankUniqueValues,
  kCurRankUniqueEmbeddings,
  // embedding shuffle
  kReverseUniqueCurRankEmbeddings,
  kReceivedEmbeddings,
  kMaxType
};

template<typename K, typename U, typename IDX>
class FusedEmbeddingTmpBufferManager final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(FusedEmbeddingTmpBufferManager);
  FusedEmbeddingTmpBufferManager(void* ptr, const int64_t num_ids, const int64_t parallel_num,
                                 bool need_process_table_ids, int64_t line_size,
                                 int64_t embedding_size, bool need_unique_values,
                                 bool need_embeddings, DataType value_dtype,
                                 DataType embedding_dtype)
      : offset_(0),
        offsets_(static_cast<size_t>(FusedEmbeddingBufferType::kMaxType), -1),
        sizes_(static_cast<size_t>(FusedEmbeddingBufferType::kMaxType)),
        ptr_(ptr) {
    // id shuffle
    const int64_t num_table_ids = need_process_table_ids ? num_ids : 0;
    const size_t table_ids_bytes = need_process_table_ids ? num_ids * sizeof(U) : 0;
    AllocBuffer(FusedEmbeddingBufferType::kNumPartitionedUnique, parallel_num * sizeof(IDX));
    size_t partitioned_ids_bytes = parallel_num * num_ids * sizeof(K);
    AllocBuffer(FusedEmbeddingBufferType::kPartitionedUniqueIds, partitioned_ids_bytes);
    AllocBuffer(FusedEmbeddingBufferType::kReceivedIds, partitioned_ids_bytes);
    AllocBuffer(FusedEmbeddingBufferType::kTableIds, table_ids_bytes);
    size_t partitioned_table_ids_bytes = parallel_num * num_table_ids * sizeof(U);
    AllocBuffer(FusedEmbeddingBufferType::kPartitionedUniqueTableIds, partitioned_table_ids_bytes);
    AllocBuffer(FusedEmbeddingBufferType::kReceivedTableIds, partitioned_table_ids_bytes);
    const size_t hash_table_capacity = parallel_num * num_ids;
    AllocBuffer(FusedEmbeddingBufferType::kWorkspace,
                hash_table_capacity * sizeof(data_shuffle::TableEntry<K>));
    size_t num_unique_matrix_bytes = parallel_num * parallel_num * sizeof(IDX);
    AllocBuffer(FusedEmbeddingBufferType::kNumUniqueMatrix, num_unique_matrix_bytes);
    size_t inverse_unique_partition_indices_bytes = num_ids * sizeof(IDX);
    AllocBuffer(FusedEmbeddingBufferType::kInverseUniquePartitionIndices,
                inverse_unique_partition_indices_bytes);
    size_t cur_rank_num_ids = parallel_num * num_ids;
    size_t cur_rank_num_table_ids = cur_rank_num_ids;
    size_t cur_rank_num_unique_bytes = sizeof(uint32_t);
    AllocBuffer(FusedEmbeddingBufferType::kCurRankNumUnique, cur_rank_num_unique_bytes);
    size_t cur_rank_unique_ids_bytes = cur_rank_num_ids * sizeof(K);
    AllocBuffer(FusedEmbeddingBufferType::kCurRankUniqueIds, cur_rank_unique_ids_bytes);
    size_t cur_rank_unique_table_ids_bytes = cur_rank_num_table_ids * sizeof(U);
    AllocBuffer(FusedEmbeddingBufferType::kCurRankUniqueTableIds, cur_rank_unique_table_ids_bytes);
    size_t cur_rank_inverse_indices_bytes = cur_rank_num_ids * sizeof(IDX);
    AllocBuffer(FusedEmbeddingBufferType::kCurRankInverseIndices, cur_rank_inverse_indices_bytes);
    // embedding lookup
    size_t num_missing_bytes = sizeof(uint32_t);
    AllocBuffer(FusedEmbeddingBufferType::kNumMissing, num_missing_bytes);
    size_t missing_indices_bytes = cur_rank_num_ids * sizeof(uint32_t);
    AllocBuffer(FusedEmbeddingBufferType::kMissingIndices, missing_indices_bytes);
    if (need_unique_values) {
      size_t cur_rank_unique_values_bytes =
          cur_rank_num_ids * line_size * GetSizeOfDataType(value_dtype);
      AllocBuffer(FusedEmbeddingBufferType::kCurRankUniqueValues, cur_rank_unique_values_bytes);
    }
    if (need_embeddings) {
      size_t cur_rank_unique_embeddings_bytes =
          cur_rank_num_ids * embedding_size * GetSizeOfDataType(embedding_dtype);
      AllocBuffer(FusedEmbeddingBufferType::kCurRankUniqueEmbeddings,
                  cur_rank_unique_embeddings_bytes);
    }
    // embedding shuffle
    size_t reverse_unique_cur_rank_embeddings_bytes =
        cur_rank_num_ids * embedding_size * GetSizeOfDataType(embedding_dtype);
    AllocBuffer(FusedEmbeddingBufferType::kReverseUniqueCurRankEmbeddings,
                reverse_unique_cur_rank_embeddings_bytes);
    size_t received_embeddings_bytes =
        cur_rank_num_ids * embedding_size * GetSizeOfDataType(embedding_dtype);
    AllocBuffer(FusedEmbeddingBufferType::kReceivedEmbeddings, received_embeddings_bytes);
  }

  template<typename T = void>
  T* Ptr(FusedEmbeddingBufferType type) const {
    CHECK(ptr_ != nullptr);
    int64_t offset = offsets_.at(static_cast<size_t>(type));
    CHECK_NE(offset, -1);
    return reinterpret_cast<T*>(reinterpret_cast<char*>(ptr_) + offset);
  }

  int64_t Size(FusedEmbeddingBufferType type) const { return sizes_.at(static_cast<size_t>(type)); }

  size_t TotalBufferSize() const { return offset_; }

 private:
  void AllocBuffer(FusedEmbeddingBufferType type, size_t size) {
    const size_t type_id = static_cast<size_t>(type);
    CHECK_EQ(offsets_.at(type_id), -1);
    offsets_.at(type_id) = offset_;
    sizes_.at(type_id) = size;
    offset_ += GetCudaAlignedSize(size);
  }
  size_t offset_;
  std::vector<int64_t> offsets_;
  std::vector<int64_t> sizes_;
  void* ptr_;
};

void MakeConstantInitializerAttr(const int64_t embedding_size, const int64_t line_size,
                                 const std::vector<float>& values, std::string* initializer_attr) {
  if (embedding_size == line_size) { return; }
  const int32_t num_states = line_size / embedding_size - 1;
  CHECK_GT(num_states, 0) << "num_states " << num_states;
  CHECK(values.size() == 0 || num_states == values.size())
      << "must set " << num_states << " optimizer states init value, but get " << values.size();
  nlohmann::json initializers;
  for (int32_t i = 0; i < num_states; ++i) {
    nlohmann::json initializer;
    initializer["type"] = "constant";
    const float initial_value = values.size() > 0 ? values.at(i) : 0.0;
    initializer["value"] = initial_value;
    initializers.push_back(initializer);
  }
  *initializer_attr = initializers.dump();
}

template<typename IDX>
class OneEmbeddingFusedLookupKernelState final : public user_op::OpKernelState {
 public:
  explicit OneEmbeddingFusedLookupKernelState(user_op::KernelInitContext* ctx)
      : device_index_(-1),
        stream_name_(EagerNcclCommMgr::kDefaultStreamName),
        parallel_desc_(ctx->parallel_desc()) {
    OF_CUDA_CHECK(cudaGetDevice(&device_index_));
    const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
    const int64_t parallel_num = ctx->parallel_ctx().parallel_num();
    OF_CUDA_CHECK(cudaMallocHost(&host_num_keys_, sizeof(IDX)));
    OF_CUDA_CHECK(
        cudaMallocHost(&host_num_unique_matrix_, parallel_num * parallel_num * sizeof(IDX)));
    const std::string& embedding_name = ctx->Attr<std::string>("embedding_name");
    key_value_store_ = Singleton<embedding::EmbeddingManager>::Get()->GetKeyValueStore(
        embedding_name, parallel_id);
    uint32_t max_query_length =
        ctx->TensorDesc4ArgNameAndIndex("ids", 0)->shape().elem_cnt() * parallel_num;
    key_value_store_->ReserveQueryLength(max_query_length);

    const int64_t embedding_size = ctx->Attr<int64_t>("embedding_size");
    const int64_t line_size = ctx->Attr<int64_t>("line_size");
    // Note(guoran): This op have no optimizer info, so set embedding states initializer constant
    // 0, which may make error in optimizer with initial_accumulator_value like adagrad and ftrl.
    std::string state_initializer;
    MakeConstantInitializerAttr(embedding_size, line_size, {}, &state_initializer);

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
  ~OneEmbeddingFusedLookupKernelState() override {
    CudaCurrentDeviceGuard guard(device_index_);
    OF_CUDA_CHECK(cudaFreeHost(host_num_keys_));
    OF_CUDA_CHECK(cudaFreeHost(host_num_unique_matrix_));
    OF_CUDA_CHECK(cudaFreeHost(host_initializer_param_));
    OF_CUDA_CHECK(cudaFree(device_initializer_param_));
    OF_CUDA_CHECK(cudaFreeHost(host_initializer_index_));
    OF_CUDA_CHECK(cudaFree(device_initializer_index_));
  }

  ncclComm_t comm() { return GetOrCreate().comm; }

  IDX* HostNumUniqueMatrix() { return host_num_unique_matrix_; }

  IDX* HostNumKeys() { return host_num_keys_; }

  embedding::KeyValueStore* KeyValueStore() { return key_value_store_; }

  const int8_t* InitializerIndex() { return device_initializer_index_; }
  const EmbeddingInitializer* Initializers() { return device_initializer_param_; }

 private:
  struct Comm {
    Comm(ncclComm_t comm) : comm(comm) {}
    ncclComm_t comm;
  };

  const Comm& GetOrCreate() {
    if (!comm_) { Init(); }
    return *comm_;
  }

  void Init() {
    std::set<std::pair<int64_t, int64_t>> device_set;
    for (int64_t parallel_id = 0; parallel_id < parallel_desc_.parallel_num(); ++parallel_id) {
      int64_t machine_id = CHECK_JUST(parallel_desc_.MachineId4ParallelId(parallel_id));
      int64_t device_id = CHECK_JUST(parallel_desc_.DeviceId4ParallelId(parallel_id));
      device_set.emplace(std::make_pair(machine_id, device_id));
    }
    EagerNcclCommMgr* comm_mgr = CHECK_NOTNULL(Singleton<EagerNcclCommMgr>::Get());
    ncclComm_t comm;
    comm = comm_mgr->GetCommForDeviceAndStreamName(device_set, stream_name_);
    comm_.reset(new Comm(comm));
  }

  int device_index_;
  std::string stream_name_;
  ParallelDesc parallel_desc_;
  std::unique_ptr<Comm> comm_;
  IDX* host_num_keys_;
  IDX* host_num_unique_matrix_;
  embedding::KeyValueStore* key_value_store_;

  EmbeddingInitializer* host_initializer_param_;
  EmbeddingInitializer* device_initializer_param_;
  int8_t* host_initializer_index_;
  int8_t* device_initializer_index_;
};

template<typename T, typename K, typename U, typename IDX>
void LookupAndInitMissing(ep::Stream* stream, OneEmbeddingFusedLookupKernelState<IDX>* kernel_state,
                          uint64_t seed, uint32_t num_unique, const int64_t embedding_size,
                          const int64_t line_size, const bool put_to_store, const void* unique_ids,
                          const void* table_ids, void* num_missing_ptr, void* missing_indices,
                          void* store_values) {
  embedding::KeyValueStore* store = kernel_state->KeyValueStore();
  const EmbeddingInitializer* initializer_param = kernel_state->Initializers();
  const int8_t* initializer_index = kernel_state->InitializerIndex();
  void* host_num_keys = kernel_state->HostNumKeys();
  LookupAndInitMissing<T, K, U, IDX>(stream, seed, store, initializer_param, initializer_index,
                                     host_num_keys, num_unique, embedding_size, line_size,
                                     put_to_store, unique_ids, table_ids, num_missing_ptr,
                                     missing_indices, store_values);
}

template<typename K, typename U, typename IDX>
void SetIdShuffleDataPtrsParam(const void* ids_ptr,
                               const FusedEmbeddingTmpBufferManager<K, U, IDX>& buffer_manager,
                               data_shuffle::IdShuffleDataPtrs<K, U, IDX>* data_ptrs) {
  data_ptrs->ids_ptr = reinterpret_cast<const K*>(ids_ptr);
  data_ptrs->table_ids_ptr = buffer_manager.template Ptr<U>(FusedEmbeddingBufferType::kTableIds);
  data_ptrs->num_partitioned_unique =
      buffer_manager.template Ptr<IDX>(FusedEmbeddingBufferType::kNumPartitionedUnique);
  data_ptrs->partitioned_unique_ids =
      buffer_manager.template Ptr<K>(FusedEmbeddingBufferType::kPartitionedUniqueIds);
  data_ptrs->partitioned_unique_table_ids =
      buffer_manager.template Ptr<U>(FusedEmbeddingBufferType::kPartitionedUniqueTableIds);
  data_ptrs->workspace_ptr = buffer_manager.Ptr(FusedEmbeddingBufferType::kWorkspace);
  data_ptrs->workspace_size = buffer_manager.Size(FusedEmbeddingBufferType::kWorkspace);
  data_ptrs->received_ids = buffer_manager.template Ptr<K>(FusedEmbeddingBufferType::kReceivedIds);
  data_ptrs->received_table_ids =
      buffer_manager.template Ptr<U>(FusedEmbeddingBufferType::kReceivedTableIds);
  data_ptrs->inverse_unique_partition_indices_ptr =
      buffer_manager.template Ptr<IDX>(FusedEmbeddingBufferType::kInverseUniquePartitionIndices);
  data_ptrs->num_unique_matrix_ptr =
      buffer_manager.template Ptr<IDX>(FusedEmbeddingBufferType::kNumUniqueMatrix);
  data_ptrs->cur_rank_num_unique_ptr =
      buffer_manager.template Ptr<IDX>(FusedEmbeddingBufferType::kCurRankNumUnique);
  data_ptrs->cur_rank_unique_ids_ptr =
      buffer_manager.template Ptr<K>(FusedEmbeddingBufferType::kCurRankUniqueIds);
  data_ptrs->cur_rank_unique_table_ids_ptr =
      buffer_manager.template Ptr<U>(FusedEmbeddingBufferType::kCurRankUniqueTableIds);
  data_ptrs->cur_rank_inverse_indices_ptr =
      buffer_manager.template Ptr<IDX>(FusedEmbeddingBufferType::kCurRankInverseIndices);
}

template<typename K, typename T, typename V, typename U, typename IDX>
class OneEmbeddingFusedLookupKernel final : public user_op::OpKernel {
 public:
  OneEmbeddingFusedLookupKernel() = default;
  ~OneEmbeddingFusedLookupKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<OneEmbeddingFusedLookupKernelState<IDX>>(ctx);
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    // IDX type is uint32_t, table_ids type is uint8_t.
    DataType num_unique_matrix_dtype = DataType::kUInt32;
    DataType table_ids_dtype = DataType::kUInt8;
    CHECK_EQ(sizeof(IDX), GetSizeOfDataType(num_unique_matrix_dtype));
    CHECK_EQ(sizeof(U), GetSizeOfDataType(table_ids_dtype));
    auto* kernel_state = dynamic_cast<OneEmbeddingFusedLookupKernelState<IDX>*>(state);
    CHECK(kernel_state != nullptr);
    const user_op::Tensor* ids = ctx->Tensor4ArgNameAndIndex("ids", 0);
    user_op::Tensor* embeddings = ctx->Tensor4ArgNameAndIndex("embeddings", 0);
    const int32_t num_tables = ctx->Attr<int32_t>("num_tables");
    // default uint8_t as table_ids type, so num_tables can not greater than 256.
    CHECK_LE(num_tables, 256) << num_tables;
    const bool has_table_ids = ctx->has_input("table_ids", 0);
    const bool need_process_table_ids = (has_table_ids || num_tables > 1);
    const int64_t num_ids = ids->shape_view().elem_cnt();
    const int64_t parallel_num = ctx->parallel_ctx().parallel_num();
    const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
    cudaStream_t cuda_stream = ctx->stream()->As<ep::CudaStream>()->cuda_stream();
    DataType value_dtype = ctx->Attr<DataType>("dtype");
    const int64_t embedding_size = ctx->Attr<int64_t>("embedding_size");
    const int64_t line_size = ctx->Attr<int64_t>("line_size");
    const int64_t padding_idx = ctx->Attr<int64_t>("padding_idx");
    const bool has_padding_idx = ctx->Attr<bool>("has_padding_idx");
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    bool need_unique_values = true;
    bool need_embeddings =
        (line_size != embedding_size) || (value_dtype != embeddings->data_type());
    FusedEmbeddingTmpBufferManager<K, U, IDX> buffer_manager(
        tmp_buffer->mut_dptr(), num_ids, parallel_num, need_process_table_ids, line_size,
        embedding_size, need_unique_values, need_embeddings, value_dtype, embeddings->data_type());
    CHECK_GE(tmp_buffer->shape_view().elem_cnt(), buffer_manager.TotalBufferSize());
    ncclComm_t comm = kernel_state->comm();
    IDX* host_num_unique_matrix = kernel_state->HostNumUniqueMatrix();
    IDX* host_num_keys = kernel_state->HostNumKeys();
    data_shuffle::IdShuffleDataPtrs<K, U, IDX> data_ptrs;
    SetIdShuffleDataPtrsParam(ids->dptr(), buffer_manager, &data_ptrs);
    // overwrite data_ptrs.table_ids_ptr
    if (need_process_table_ids) {
      U* tmp_table_ids_ptr = buffer_manager.template Ptr<U>(FusedEmbeddingBufferType::kTableIds);
      data_ptrs.table_ids_ptr = tmp_table_ids_ptr;
      if (has_table_ids) {
        // use table_id default data_type uint8, if has input table_ids with different data_type,
        // cast it to uint8.
        const user_op::Tensor* table_ids = ctx->Tensor4ArgNameAndIndex("table_ids", 0);
        if (table_ids->data_type() != table_ids_dtype) {
          std::unique_ptr<ep::primitive::Cast> cast_primitive =
              ep::primitive::NewPrimitive<ep::primitive::CastFactory>(
                  DeviceType::kCUDA, table_ids->data_type(), table_ids_dtype);
          cast_primitive->Launch(ctx->stream(), table_ids->dptr(), tmp_table_ids_ptr,
                                 table_ids->shape_view().elem_cnt());
        } else {
          data_ptrs.table_ids_ptr = reinterpret_cast<const U*>(table_ids->dptr());
        }
      } else {
        const int32_t num_tables = ctx->Attr<int32_t>("num_tables");
        data_shuffle::GenerateTableIds<<<BlocksNum4ThreadsNum(num_ids), kCudaThreadsNumPerBlock, 0,
                                         cuda_stream>>>(num_ids, num_tables, tmp_table_ids_ptr);
      }
    } else {
      data_ptrs.table_ids_ptr = nullptr;
    }

    data_shuffle::IdShuffle(ctx->stream(), comm, data_ptrs, num_ids, parallel_id, parallel_num,
                            num_unique_matrix_dtype, ids->data_type(), table_ids_dtype,
                            need_process_table_ids, has_padding_idx, padding_idx,
                            host_num_unique_matrix, host_num_keys);
    uint32_t num_unique = *host_num_keys;

    // lookup and put, if is_full_cache, not put to store.
    uint32_t* num_missing_ptr =
        buffer_manager.template Ptr<uint32_t>(FusedEmbeddingBufferType::kNumMissing);
    uint32_t* missing_indices_ptr =
        buffer_manager.template Ptr<uint32_t>(FusedEmbeddingBufferType::kMissingIndices);
    void* values_ptr =
        buffer_manager.template Ptr<V>(FusedEmbeddingBufferType::kCurRankUniqueValues);
    T* cur_rank_embeddings_ptr =
        need_embeddings
            ? buffer_manager.template Ptr<T>(FusedEmbeddingBufferType::kCurRankUniqueEmbeddings)
            : reinterpret_cast<T*>(values_ptr);
    const bool is_full_cache = ctx->Attr<bool>("is_full_cache");
    const bool put_to_store = (!is_full_cache);
    const int64_t seed = ctx->Attr<int64_t>("seed");
    LookupAndInitMissing<V, K, U, IDX>(
        ctx->stream(), kernel_state, seed, num_unique, embedding_size, line_size, put_to_store,
        data_ptrs.cur_rank_unique_ids_ptr, data_ptrs.cur_rank_unique_table_ids_ptr, num_missing_ptr,
        missing_indices_ptr, values_ptr);
    if (need_embeddings) {
      CopyValuesToEmbeddings<V>(ctx->stream(), num_unique, embedding_size, line_size, value_dtype,
                                embeddings->data_type(), reinterpret_cast<V*>(values_ptr),
                                cur_rank_embeddings_ptr);
    }

    // embedding shuffle
    int64_t cur_rank_num_ids = 0;
    for (int64_t i = 0; i < parallel_num; ++i) {
      cur_rank_num_ids += host_num_unique_matrix[i * parallel_num + parallel_id];
    }
    int64_t unique_partitioned_num_ids = 0;
    for (int64_t i = 0; i < parallel_num; ++i) {
      unique_partitioned_num_ids += host_num_unique_matrix[parallel_id * parallel_num + i];
    }
    T* reverse_unique_cur_rank_embeddings_ptr =
        buffer_manager.template Ptr<T>(FusedEmbeddingBufferType::kReverseUniqueCurRankEmbeddings);
    T* received_embeddings_ptr =
        buffer_manager.template Ptr<T>(FusedEmbeddingBufferType::kReceivedEmbeddings);
    GatherKernelUtilImpl<DeviceType::kCUDA, T, IDX>::Forward(
        ctx->stream(), data_ptrs.cur_rank_inverse_indices_ptr, cur_rank_num_ids,
        cur_rank_embeddings_ptr, Shape({1, num_unique, embedding_size}),
        reverse_unique_cur_rank_embeddings_ptr, 0);

    data_shuffle::ShuffleEmbeddings(cuda_stream, comm, parallel_id, parallel_num, num_ids,
                                    embedding_size, embeddings->data_type(), host_num_unique_matrix,
                                    reverse_unique_cur_rank_embeddings_ptr,
                                    received_embeddings_ptr);
    GatherKernelUtilImpl<DeviceType::kCUDA, T, IDX>::Forward(
        ctx->stream(), data_ptrs.inverse_unique_partition_indices_ptr, num_ids,
        received_embeddings_ptr, Shape({1, unique_partitioned_num_ids, embedding_size}),
        embeddings->mut_dptr<T>(), 0);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

auto SingleDeviceKernel() {
  return hob::make_custom("SingleDeviceKernel", [](const user_op::KernelRegContext& ctx) {
    return (ctx.parallel_ctx().parallel_num() == 1);
  });
}

// Note(guoran): Default use U type as uint8_t, IDX as uint32_t. Because table_ids is optional, so
// can not use it in hob, if has table_ids input and dtype is not uint8_t cast to uint8_t in kernel.
#define REGISTER_CUDA_ONE_EMBEDDING_FUSED_LOOKUP_KERNEL(k_dtype_pair, t_dtype_pair, v_dtype_pair) \
  REGISTER_USER_KERNEL("one_embedding_fused_lookup")                                              \
      .SetCreateFn<OneEmbeddingFusedLookupKernel<                                                 \
          OF_PP_PAIR_FIRST(k_dtype_pair), OF_PP_PAIR_FIRST(t_dtype_pair),                         \
          OF_PP_PAIR_FIRST(v_dtype_pair), uint8_t, uint32_t>>()                                   \
      .SetIsMatchedHob(                                                                           \
          (user_op::HobDeviceType() == DeviceType::kCUDA)                                         \
          && (user_op::HobDataType("ids", 0) == OF_PP_PAIR_SECOND(k_dtype_pair))                  \
          && (user_op::HobDataType("embeddings", 0) == OF_PP_PAIR_SECOND(t_dtype_pair))           \
          && (user_op::HobAttr<DataType>("dtype") == OF_PP_PAIR_SECOND(v_dtype_pair))             \
          && !SingleDeviceKernel())                                                               \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                         \
        const user_op::TensorDesc& ids = ctx->InputTensorDesc("ids", 0);                          \
        const user_op::TensorDesc& embeddings = ctx->OutputTensorDesc("embeddings", 0);           \
        const bool has_table_ids = ctx->has_input("table_ids", 0);                                \
        const int32_t num_tables = ctx->Attr<int32_t>("num_tables");                              \
        const bool need_process_table_ids = (has_table_ids || num_tables > 1);                    \
        DataType value_dtype = ctx->Attr<DataType>("dtype");                                      \
        const int64_t embedding_size = ctx->Attr<int64_t>("embedding_size");                      \
        const int64_t line_size = ctx->Attr<int64_t>("line_size");                                \
        bool need_embeddings =                                                                    \
            (line_size != embedding_size) || (value_dtype != embeddings.data_type());             \
        FusedEmbeddingTmpBufferManager<OF_PP_PAIR_FIRST(k_dtype_pair), uint8_t, uint32_t>         \
            buffer_manager(nullptr, ids.shape().elem_cnt(), ctx->parallel_ctx().parallel_num(),   \
                           need_process_table_ids, line_size, embedding_size, true,               \
                           need_embeddings, value_dtype, embeddings.data_type());                 \
        return buffer_manager.TotalBufferSize();                                                  \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_CUDA_ONE_EMBEDDING_FUSED_LOOKUP_KERNEL, ID_DATA_TYPE_SEQ,
                                 FLOATING_DATA_TYPE_SEQ HALF_DATA_TYPE_SEQ, EMBEDDING_DATA_TYPE_SEQ)

template<typename IDX>
class OneEmbeddingFusedLookupLocalKernelState final : public user_op::OpKernelState {
 public:
  explicit OneEmbeddingFusedLookupLocalKernelState(user_op::KernelInitContext* ctx)
      : device_index_(-1) {
    OF_CUDA_CHECK(cudaGetDevice(&device_index_));
    const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
    OF_CUDA_CHECK(cudaMallocHost(&host_num_keys_, sizeof(IDX)));
    const std::string& embedding_name = ctx->Attr<std::string>("embedding_name");
    key_value_store_ = Singleton<embedding::EmbeddingManager>::Get()->GetKeyValueStore(
        embedding_name, parallel_id);
    uint32_t max_query_length = ctx->TensorDesc4ArgNameAndIndex("ids", 0)->shape().elem_cnt();
    key_value_store_->ReserveQueryLength(max_query_length);

    const int64_t embedding_size = ctx->Attr<int64_t>("embedding_size");
    const int64_t line_size = ctx->Attr<int64_t>("line_size");
    // Note(guoran): This op have no optimizer info, so set embedding states initializer constant
    // 0, which may make error in optimizer with initial_accumulator_value like adagrad and ftrl.
    std::string state_initializer;
    MakeConstantInitializerAttr(embedding_size, line_size, {}, &state_initializer);

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
  ~OneEmbeddingFusedLookupLocalKernelState() override {
    CudaCurrentDeviceGuard guard(device_index_);
    OF_CUDA_CHECK(cudaFreeHost(host_num_keys_));
    OF_CUDA_CHECK(cudaFreeHost(host_initializer_param_));
    OF_CUDA_CHECK(cudaFree(device_initializer_param_));
    OF_CUDA_CHECK(cudaFreeHost(host_initializer_index_));
    OF_CUDA_CHECK(cudaFree(device_initializer_index_));
  }

  IDX* HostNumKeys() { return host_num_keys_; }

  embedding::KeyValueStore* KeyValueStore() { return key_value_store_; }

  const int8_t* InitializerIndex() { return device_initializer_index_; }
  const EmbeddingInitializer* Initializers() { return device_initializer_param_; }

 private:
  int device_index_;
  IDX* host_num_keys_;
  embedding::KeyValueStore* key_value_store_;

  EmbeddingInitializer* host_initializer_param_;
  EmbeddingInitializer* device_initializer_param_;
  int8_t* host_initializer_index_;
  int8_t* device_initializer_index_;
};

template<typename T, typename K, typename U, typename IDX>
void LookupAndInitMissing(ep::Stream* stream,
                          OneEmbeddingFusedLookupLocalKernelState<IDX>* kernel_state, uint64_t seed,
                          uint32_t num_unique, const int64_t embedding_size,
                          const int64_t line_size, const bool put_to_store, const void* unique_ids,
                          const void* table_ids, void* num_missing_ptr, void* missing_indices,
                          void* store_values) {
  embedding::KeyValueStore* store = kernel_state->KeyValueStore();
  const EmbeddingInitializer* initializer_param = kernel_state->Initializers();
  const int8_t* initializer_index = kernel_state->InitializerIndex();
  void* host_num_keys = kernel_state->HostNumKeys();
  LookupAndInitMissing<T, K, U, IDX>(stream, seed, store, initializer_param, initializer_index,
                                     host_num_keys, num_unique, embedding_size, line_size,
                                     put_to_store, unique_ids, table_ids, num_missing_ptr,
                                     missing_indices, store_values);
}

template<typename K, typename U, typename IDX>
class FusedLocalEmbeddingTmpBufferManager final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(FusedLocalEmbeddingTmpBufferManager);
  FusedLocalEmbeddingTmpBufferManager(void* ptr, const int64_t num_ids, bool need_process_table_ids,
                                      int64_t line_size, int64_t embedding_size,
                                      bool need_embeddings, DataType value_dtype,
                                      DataType embedding_dtype)
      : offset_(0),
        offsets_(static_cast<size_t>(FusedEmbeddingBufferType::kMaxType), -1),
        sizes_(static_cast<size_t>(FusedEmbeddingBufferType::kMaxType)),
        ptr_(ptr) {
    // id shuffle
    const size_t table_ids_bytes = need_process_table_ids ? num_ids * sizeof(U) : 0;
    AllocBuffer(FusedEmbeddingBufferType::kTableIds, table_ids_bytes);
    const size_t hash_table_capacity = num_ids;
    AllocBuffer(FusedEmbeddingBufferType::kWorkspace,
                hash_table_capacity * sizeof(data_shuffle::TableEntry<K>));
    size_t cur_rank_num_ids = num_ids;
    size_t cur_rank_num_table_ids = cur_rank_num_ids;
    size_t cur_rank_num_unique_bytes = sizeof(uint32_t);
    AllocBuffer(FusedEmbeddingBufferType::kCurRankNumUnique, cur_rank_num_unique_bytes);
    size_t cur_rank_unique_ids_bytes = cur_rank_num_ids * sizeof(K);
    AllocBuffer(FusedEmbeddingBufferType::kCurRankUniqueIds, cur_rank_unique_ids_bytes);
    size_t cur_rank_unique_table_ids_bytes = cur_rank_num_table_ids * sizeof(U);
    AllocBuffer(FusedEmbeddingBufferType::kCurRankUniqueTableIds, cur_rank_unique_table_ids_bytes);
    size_t cur_rank_inverse_indices_bytes = cur_rank_num_ids * sizeof(IDX);
    AllocBuffer(FusedEmbeddingBufferType::kCurRankInverseIndices, cur_rank_inverse_indices_bytes);
    // embedding lookup
    size_t num_missing_bytes = sizeof(uint32_t);
    AllocBuffer(FusedEmbeddingBufferType::kNumMissing, num_missing_bytes);
    size_t missing_indices_bytes = cur_rank_num_ids * sizeof(uint32_t);
    AllocBuffer(FusedEmbeddingBufferType::kMissingIndices, missing_indices_bytes);
    size_t cur_rank_unique_values_bytes =
        cur_rank_num_ids * line_size * GetSizeOfDataType(value_dtype);
    AllocBuffer(FusedEmbeddingBufferType::kCurRankUniqueValues, cur_rank_unique_values_bytes);
    if (need_embeddings) {
      size_t cur_rank_unique_embeddings_bytes =
          cur_rank_num_ids * embedding_size * GetSizeOfDataType(embedding_dtype);
      AllocBuffer(FusedEmbeddingBufferType::kCurRankUniqueEmbeddings,
                  cur_rank_unique_embeddings_bytes);
    }
  }

  template<typename T = void>
  T* Ptr(FusedEmbeddingBufferType type) const {
    CHECK(ptr_ != nullptr);
    int64_t offset = offsets_.at(static_cast<size_t>(type));
    CHECK_NE(offset, -1);
    return reinterpret_cast<T*>(reinterpret_cast<char*>(ptr_) + offset);
  }

  int64_t Size(FusedEmbeddingBufferType type) const { return sizes_.at(static_cast<size_t>(type)); }

  size_t TotalBufferSize() const { return offset_; }

 private:
  void AllocBuffer(FusedEmbeddingBufferType type, size_t size) {
    const size_t type_id = static_cast<size_t>(type);
    CHECK_EQ(offsets_.at(type_id), -1);
    offsets_.at(type_id) = offset_;
    sizes_.at(type_id) = size;
    offset_ += GetCudaAlignedSize(size);
  }
  size_t offset_;
  std::vector<int64_t> offsets_;
  std::vector<int64_t> sizes_;
  void* ptr_;
};

template<typename K, typename T, typename V, typename U, typename IDX>
class OneEmbeddingFusedLookupLocalKernel final : public user_op::OpKernel {
 public:
  OneEmbeddingFusedLookupLocalKernel() = default;
  ~OneEmbeddingFusedLookupLocalKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<OneEmbeddingFusedLookupLocalKernelState<IDX>>(ctx);
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    // IDX type is uint32_t, table_ids type is uint8_t.
    DataType num_unique_matrix_dtype = DataType::kUInt32;
    DataType table_ids_dtype = DataType::kUInt8;
    CHECK_EQ(sizeof(IDX), GetSizeOfDataType(num_unique_matrix_dtype));
    CHECK_EQ(sizeof(U), GetSizeOfDataType(table_ids_dtype));
    auto* kernel_state = dynamic_cast<OneEmbeddingFusedLookupLocalKernelState<IDX>*>(state);
    CHECK(kernel_state != nullptr);
    const user_op::Tensor* ids = ctx->Tensor4ArgNameAndIndex("ids", 0);
    user_op::Tensor* embeddings = ctx->Tensor4ArgNameAndIndex("embeddings", 0);
    const int32_t num_tables = ctx->Attr<int32_t>("num_tables");
    // default uint8_t as table_ids type, so num_tables can not greater than 256.
    CHECK_LE(num_tables, 256) << num_tables;
    const bool has_table_ids = ctx->has_input("table_ids", 0);
    const bool need_process_table_ids = (has_table_ids || num_tables > 1);
    const int64_t num_ids = ids->shape_view().elem_cnt();
    cudaStream_t cuda_stream = ctx->stream()->As<ep::CudaStream>()->cuda_stream();
    DataType value_dtype = ctx->Attr<DataType>("dtype");
    const int64_t embedding_size = ctx->Attr<int64_t>("embedding_size");
    const int64_t line_size = ctx->Attr<int64_t>("line_size");
    const int64_t padding_idx = ctx->Attr<int64_t>("padding_idx");
    const bool has_padding_idx = ctx->Attr<bool>("has_padding_idx");
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    bool need_embeddings =
        (line_size != embedding_size) || (value_dtype != embeddings->data_type());
    FusedLocalEmbeddingTmpBufferManager<K, U, IDX> buffer_manager(
        tmp_buffer->mut_dptr(), num_ids, need_process_table_ids, line_size, embedding_size,
        need_embeddings, value_dtype, embeddings->data_type());
    CHECK_GE(tmp_buffer->shape_view().elem_cnt(), buffer_manager.TotalBufferSize());
    IDX* host_num_keys = kernel_state->HostNumKeys();

    const U* table_ids_ptr = nullptr;
    if (need_process_table_ids) {
      U* tmp_table_ids_ptr = buffer_manager.template Ptr<U>(FusedEmbeddingBufferType::kTableIds);
      table_ids_ptr = tmp_table_ids_ptr;
      if (has_table_ids) {
        // use table_id default data_type uint8, if has input table_ids with different data_type,
        // cast it to uint8.
        const user_op::Tensor* table_ids = ctx->Tensor4ArgNameAndIndex("table_ids", 0);
        if (table_ids->data_type() != table_ids_dtype) {
          std::unique_ptr<ep::primitive::Cast> cast_primitive =
              ep::primitive::NewPrimitive<ep::primitive::CastFactory>(
                  DeviceType::kCUDA, table_ids->data_type(), table_ids_dtype);
          cast_primitive->Launch(ctx->stream(), table_ids->dptr(), tmp_table_ids_ptr,
                                 table_ids->shape_view().elem_cnt());
        } else {
          table_ids_ptr = reinterpret_cast<const U*>(table_ids->dptr());
        }
      } else {
        const int32_t num_tables = ctx->Attr<int32_t>("num_tables");
        data_shuffle::GenerateTableIds<<<BlocksNum4ThreadsNum(num_ids), kCudaThreadsNumPerBlock, 0,
                                         cuda_stream>>>(num_ids, num_tables, tmp_table_ids_ptr);
      }
    }
    IDX* num_unique_ptr =
        buffer_manager.template Ptr<IDX>(FusedEmbeddingBufferType::kCurRankNumUnique);
    K* unique_ids_ptr = buffer_manager.template Ptr<K>(FusedEmbeddingBufferType::kCurRankUniqueIds);
    U* unique_table_ids_ptr =
        buffer_manager.template Ptr<U>(FusedEmbeddingBufferType::kCurRankUniqueTableIds);
    IDX* inverse_indices_ptr =
        buffer_manager.template Ptr<IDX>(FusedEmbeddingBufferType::kCurRankInverseIndices);
    void* workspace_ptr = buffer_manager.Ptr(FusedEmbeddingBufferType::kWorkspace);
    const size_t workspace_bytes = buffer_manager.Size(FusedEmbeddingBufferType::kWorkspace);
    int64_t hash_capacity = num_ids;
    data_shuffle::UniqueAndPartition<K, U, IDX, embedding::GlobalUniqueHash>(
        cuda_stream, num_ids, hash_capacity, 1, reinterpret_cast<const K*>(ids->dptr()),
        table_ids_ptr, num_unique_ptr, unique_ids_ptr, unique_table_ids_ptr, inverse_indices_ptr,
        reinterpret_cast<data_shuffle::TableEntry<K>*>(workspace_ptr), workspace_bytes,
        need_process_table_ids, has_padding_idx, padding_idx);

    OF_CUDA_CHECK(cudaMemcpyAsync(host_num_keys, num_unique_ptr, sizeof(IDX), cudaMemcpyDefault,
                                  cuda_stream));
    CHECK_JUST(ctx->stream()->Sync());

    uint32_t num_unique = *host_num_keys;

    // lookup and put, if is_full_cache, not put to store.
    uint32_t* num_missing_ptr =
        buffer_manager.template Ptr<uint32_t>(FusedEmbeddingBufferType::kNumMissing);
    uint32_t* missing_indices_ptr =
        buffer_manager.template Ptr<uint32_t>(FusedEmbeddingBufferType::kMissingIndices);
    void* values_ptr =
        buffer_manager.template Ptr<V>(FusedEmbeddingBufferType::kCurRankUniqueValues);
    T* cur_rank_embeddings_ptr =
        need_embeddings
            ? buffer_manager.template Ptr<T>(FusedEmbeddingBufferType::kCurRankUniqueEmbeddings)
            : reinterpret_cast<T*>(values_ptr);
    const bool is_full_cache = ctx->Attr<bool>("is_full_cache");
    const bool put_to_store = (!is_full_cache);
    const int64_t seed = ctx->Attr<int64_t>("seed");
    LookupAndInitMissing<V, K, U, IDX>(
        ctx->stream(), kernel_state, seed, num_unique, embedding_size, line_size, put_to_store,
        unique_ids_ptr, unique_table_ids_ptr, num_missing_ptr, missing_indices_ptr, values_ptr);
    if (need_embeddings) {
      CopyValuesToEmbeddings<V>(ctx->stream(), num_unique, embedding_size, line_size, value_dtype,
                                embeddings->data_type(), reinterpret_cast<V*>(values_ptr),
                                cur_rank_embeddings_ptr);
    }
    // gather
    GatherKernelUtilImpl<DeviceType::kCUDA, T, IDX>::Forward(
        ctx->stream(), inverse_indices_ptr, num_ids, cur_rank_embeddings_ptr,
        Shape({1, num_unique, embedding_size}), embeddings->mut_dptr<T>(), 0);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

// Note(guoran): Default use U type as uint8_t, IDX as uint32_t. Because table_ids is optional, so
// can not use it in hob, if has table_ids input and dtype is not uint8_t cast to uint8_t in kernel.
#define REGISTER_CUDA_ONE_EMBEDDING_FUSED_LOOKUP_LOCAL_KERNEL(k_dtype_pair, t_dtype_pair,         \
                                                              v_dtype_pair)                       \
  REGISTER_USER_KERNEL("one_embedding_fused_lookup")                                              \
      .SetCreateFn<OneEmbeddingFusedLookupLocalKernel<                                            \
          OF_PP_PAIR_FIRST(k_dtype_pair), OF_PP_PAIR_FIRST(t_dtype_pair),                         \
          OF_PP_PAIR_FIRST(v_dtype_pair), uint8_t, uint32_t>>()                                   \
      .SetIsMatchedHob(                                                                           \
          (user_op::HobDeviceType() == DeviceType::kCUDA)                                         \
          && (user_op::HobDataType("ids", 0) == OF_PP_PAIR_SECOND(k_dtype_pair))                  \
          && (user_op::HobDataType("embeddings", 0) == OF_PP_PAIR_SECOND(t_dtype_pair))           \
          && (user_op::HobAttr<DataType>("dtype") == OF_PP_PAIR_SECOND(v_dtype_pair))             \
          && SingleDeviceKernel())                                                                \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                         \
        const user_op::TensorDesc& ids = ctx->InputTensorDesc("ids", 0);                          \
        const user_op::TensorDesc& embeddings = ctx->OutputTensorDesc("embeddings", 0);           \
        const bool has_table_ids = ctx->has_input("table_ids", 0);                                \
        const int32_t num_tables = ctx->Attr<int32_t>("num_tables");                              \
        const bool need_process_table_ids = (has_table_ids || num_tables > 1);                    \
        DataType value_dtype = ctx->Attr<DataType>("dtype");                                      \
        const int64_t embedding_size = ctx->Attr<int64_t>("embedding_size");                      \
        const int64_t line_size = ctx->Attr<int64_t>("line_size");                                \
        bool need_embeddings =                                                                    \
            (line_size != embedding_size) || (value_dtype != embeddings.data_type());             \
        FusedLocalEmbeddingTmpBufferManager<OF_PP_PAIR_FIRST(k_dtype_pair), uint8_t, uint32_t>    \
            buffer_manager(nullptr, ids.shape().elem_cnt(), need_process_table_ids, line_size,    \
                           embedding_size, need_embeddings, value_dtype, embeddings.data_type()); \
        return buffer_manager.TotalBufferSize();                                                  \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_CUDA_ONE_EMBEDDING_FUSED_LOOKUP_LOCAL_KERNEL,
                                 ID_DATA_TYPE_SEQ, FLOATING_DATA_TYPE_SEQ HALF_DATA_TYPE_SEQ,
                                 EMBEDDING_DATA_TYPE_SEQ)

class OneEmbeddingFusedLookupGradKernel final : public user_op::OpKernel {
 public:
  OneEmbeddingFusedLookupGradKernel() = default;
  ~OneEmbeddingFusedLookupGradKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    // do nothing
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("one_embedding_fused_lookup_grad")
    .SetCreateFn<OneEmbeddingFusedLookupGradKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA));

}  // namespace oneflow
