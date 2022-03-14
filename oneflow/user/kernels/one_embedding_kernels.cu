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
#include "oneflow/user/kernels/random_mask_generator.h"
#include "oneflow/core/framework/random_generator_impl.h"
#include "oneflow/core/cuda/atomic.cuh"
#include "oneflow/core/ep/include/primitive/copy_nd.h"
#include "oneflow/core/ep/include/primitive/cast.h"
#include "nlohmann/json.hpp"

namespace oneflow {

namespace {

enum class InitializerType {
  kUniform,
  kNormal,
};

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
  };
};

struct EmbeddingColumn {
  EmbeddingInitializer initializer;
};

constexpr size_t kMaxColumns = 128;

struct ColumnsParam {
  int32_t num_columns = 0;
  EmbeddingColumn columns[kMaxColumns];
};

void ParseColumnFromJson(const nlohmann::json& initializer, EmbeddingColumn* embedding_column) {
  CHECK(initializer.contains("type"));
  CHECK(initializer["type"].is_string());
  std::string type = initializer["type"].get<std::string>();
  if (type == "uniform") {
    embedding_column->initializer.type = InitializerType::kUniform;
    CHECK(initializer.contains("low"));
    CHECK(initializer.contains("high"));
    CHECK(initializer["low"].is_number());
    CHECK(initializer["high"].is_number());
    embedding_column->initializer.uniform_param.low = initializer["low"];
    embedding_column->initializer.uniform_param.high = initializer["high"];
  } else if (type == "normal") {
    CHECK(initializer.contains("mean"));
    CHECK(initializer.contains("std"));
    CHECK(initializer["mean"].is_number());
    CHECK(initializer["std"].is_number());
    embedding_column->initializer.type = InitializerType::kNormal;
    embedding_column->initializer.normal_param.mean = initializer["mean"];
    embedding_column->initializer.normal_param.std = initializer["std"];
  } else {
    UNIMPLEMENTED();
  }
}

class EmbeddingColumns {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EmbeddingColumns);
  EmbeddingColumns(std::string json_serialized) {
    auto json_object = nlohmann::json::parse(json_serialized);
    CHECK(json_object.contains("columns"));
    auto columns = json_object["columns"];
    CHECK(columns.is_array());
    CHECK_LE(columns.size(), kMaxColumns);
    for (int32_t i = 0; i < columns.size(); ++i) {
      auto column = columns.at(i);
      CHECK(column.contains("initializer"));
      ParseColumnFromJson(column["initializer"], &(param_.columns[i]));
    }
    param_.num_columns = columns.size();
  }
  ColumnsParam Columns() const { return param_; }

 private:
  ColumnsParam param_;
};

template<typename IDX>
class EmbeddingKernelState final : public user_op::OpKernelState {
 public:
  explicit EmbeddingKernelState(user_op::KernelInitContext* ctx)
      : device_index_(-1), generator_(CHECK_JUST(one::MakeGenerator(DeviceType::kCUDA))) {
    OF_CUDA_CHECK(cudaGetDevice(&device_index_));
    OF_CUDA_CHECK(cudaMallocHost(&host_num_keys_, sizeof(IDX)));
    EmbeddingColumns embedding_columns(ctx->Attr<std::string>("embedding_columns"));
    uint32_t max_query_length =
        ctx->TensorDesc4ArgNameAndIndex("unique_ids", 0)->shape().elem_cnt();
    // TODO: depend on EmbeddingManager
    // key_value_store_ = Global<EmbeddingManager>::Get()->GetKeyValueStore(
    //    ctx->Attr<std::string>("embedding_name"), ctx->parallel_ctx().parallel_id());
    // key_value_store_->ReserveQueryLength(max_query_length);
    columns_param_ = embedding_columns.Columns();
  }
  ~EmbeddingKernelState() {
    CudaCurrentDeviceGuard guard(device_index_);
    OF_CUDA_CHECK(cudaFreeHost(host_num_keys_));
  }

  void* HostNumKeys() { return host_num_keys_; }

  embedding::KeyValueStore* KeyValueStore() { return key_value_store_; }

  one::Generator* generator() { return generator_.get(); }

  ColumnsParam Columns() { return columns_param_; }

 private:
  int device_index_;
  void* host_num_keys_;
  std::shared_ptr<one::Generator> generator_;
  embedding::KeyValueStore* key_value_store_;
  ColumnsParam columns_param_;
};

template<typename IDX>
class EmbeddingPutKernelState final : public user_op::OpKernelState {
 public:
  explicit EmbeddingPutKernelState(user_op::KernelInitContext* ctx) : device_index_(-1) {
    OF_CUDA_CHECK(cudaGetDevice(&device_index_));
    OF_CUDA_CHECK(cudaMallocHost(&host_num_keys_, sizeof(IDX)));
    uint32_t max_query_length =
        ctx->TensorDesc4ArgNameAndIndex("unique_ids", 0)->shape().elem_cnt();
    // TODO: depend on EmbeddingManager
    // key_value_store_ = Global<EmbeddingManager>::Get()->GetKeyValueStore(
    //    ctx->Attr<std::string>("embedding_name"), ctx->parallel_ctx().parallel_id());
    // key_value_store_->ReserveQueryLength(max_query_length);
  }
  ~EmbeddingPutKernelState() {
    CudaCurrentDeviceGuard guard(device_index_);
    OF_CUDA_CHECK(cudaFreeHost(host_num_keys_));
  }

  void* HostNumKeys() { return host_num_keys_; }
  embedding::KeyValueStore* KeyValueStore() { return key_value_store_; }

 private:
  int device_index_;
  void* host_num_keys_;
  embedding::KeyValueStore* key_value_store_;
};

enum class EmbeddingBufferType { kNumMissing = 0, kMissingIndices, kValues, kMaxType };

class EmbeddingTmpBufferManager final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EmbeddingTmpBufferManager);
  EmbeddingTmpBufferManager(void* ptr, const int64_t num_ids, const int64_t value_byte_size,
                            const bool need_value_buffer)
      : offset_(0), offsets_(static_cast<size_t>(EmbeddingBufferType::kMaxType), -1), ptr_(ptr) {
    AllocBuffer(EmbeddingBufferType::kNumMissing, sizeof(uint32_t));
    AllocBuffer(EmbeddingBufferType::kMissingIndices, num_ids * sizeof(uint32_t));
    if (need_value_buffer) { AllocBuffer(EmbeddingBufferType::kValues, num_ids * value_byte_size); }
  }

  template<typename T = void>
  T* Ptr(EmbeddingBufferType type) {
    CHECK(ptr_ != nullptr);
    int64_t offset = offsets_.at(static_cast<size_t>(type));
    CHECK_NE(offset, -1);
    return reinterpret_cast<T*>(reinterpret_cast<char*>(ptr_) + offset);
  }

  size_t TotalBufferSize() const { return offset_; }

 private:
  void AllocBuffer(EmbeddingBufferType type, size_t size) {
    const size_t type_id = static_cast<size_t>(type);
    CHECK_EQ(offsets_.at(type_id), -1);
    offsets_.at(type_id) = offset_;
    offset_ += GetCudaAlignedSize(size);
  }

  size_t offset_;
  std::vector<int64_t> offsets_;
  void* ptr_;
};

template<typename T, typename U>
__global__ void InitValueKernel(uint64_t seed, one::CUDAGeneratorState* cuda_gen_state,
                                uint64_t inc_offset, const int32_t line_size,
                                const int32_t embedding_size, ColumnsParam param,
                                const U* column_ids, const uint32_t* num_missing_keys,
                                const uint32_t* missing_indices, T* values) {
  int32_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, global_thread_id, cuda_gen_state->dev_offset, &state);
  int64_t n = *num_missing_keys * line_size;
  CUDA_1D_KERNEL_LOOP(i, n) {
    int row = i / line_size;
    int col = i - row * line_size;
    const uint32_t index = missing_indices[row];
    const int64_t offset = index * line_size + col;
    const int32_t slot_idx = column_ids[index];
    assert(slot_idx < param.num_columns);
    EmbeddingInitializer initializer = param.columns[slot_idx].initializer;
    T value = 0;
    if (col < embedding_size) {
      if (initializer.type == InitializerType::kUniform) {
        const float low = initializer.uniform_param.low;
        const float high = initializer.uniform_param.high;
        value = curand_uniform(&state) * (high - low) + low;
      } else if (initializer.type == InitializerType::kNormal) {
        const float mean = initializer.normal_param.mean;
        const float std = initializer.normal_param.std;
        value = (curand_normal(&state) + mean) / std;
      } else {
        __trap();
      }
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
                          const int64_t num_ids, const int32_t embedding_size,
                          const int32_t line_size, const void* num_unique_ptr,
                          const void* unique_ids, const void* column_ids, T* values_ptr,
                          void* tmp_buffer_ptr, uint32_t* return_num_unique,
                          const bool put_to_kv_store) {
  const auto& generator = kernel_state->generator();
  CHECK_NOTNULL(generator);
  std::shared_ptr<one::CUDAGeneratorImpl> cuda_generator =
      CHECK_JUST(generator->template Get<one::CUDAGeneratorImpl>());
  uint64_t seed = cuda_generator->current_seed();
  one::CUDAGeneratorState* cuda_gen_state = cuda_generator->cuda_gen_state();
  embedding::KeyValueStore* store = kernel_state->KeyValueStore();
  ColumnsParam param = kernel_state->Columns();
  bool need_value_buffer;
  if (values_ptr != nullptr) {
    need_value_buffer = false;
  } else {
    need_value_buffer = true;
  }
  EmbeddingTmpBufferManager buffer_manager(tmp_buffer_ptr, num_ids, line_size * sizeof(T),
                                           need_value_buffer);
  void* host_num_keys = kernel_state->HostNumKeys();
  OF_CUDA_CHECK(cudaMemcpyAsync(host_num_keys, num_unique_ptr, sizeof(IDX), cudaMemcpyDefault,
                                stream->As<ep::CudaStream>()->cuda_stream()));
  CHECK_JUST(stream->Sync());
  uint32_t num_unique = *reinterpret_cast<IDX*>(host_num_keys);
  uint32_t* num_missing_ptr =
      buffer_manager.template Ptr<uint32_t>(EmbeddingBufferType::kNumMissing);
  uint32_t* missing_indices =
      buffer_manager.template Ptr<uint32_t>(EmbeddingBufferType::kMissingIndices);
  T* store_values =
      need_value_buffer ? buffer_manager.template Ptr<T>(EmbeddingBufferType::kValues) : values_ptr;
  store->Get(stream, num_unique, unique_ids, store_values, num_missing_ptr, missing_indices);
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
            seed, cuda_gen_state, inc_offset, line_size, embedding_size, param,
            reinterpret_cast<const U*>(column_ids), num_missing_ptr, missing_indices, store_values);
  }
  if (put_to_kv_store) { store->Put(stream, num_unique, unique_ids, store_values); }
  *return_num_unique = num_unique;
}

template<typename T, typename U>
__global__ void CastAndCopyNd(int64_t out_elem_cnt, const int32_t in_cols, const int32_t out_cols,
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
      CastAndCopyNd<T, half><<<BlocksNum4ThreadsNum(embedding_elem_cnt), kCudaThreadsNumPerBlock, 0,
                               stream->As<ep::CudaStream>()->cuda_stream()>>>(
          embedding_elem_cnt, value_size, embedding_size, values,
          reinterpret_cast<half*>(embeddings));
    } else {
      UNIMPLEMENTED();
    }
  }
}

}  // namespace

template<typename T, typename U, typename IDX>
class EmbeddingPrefetchKernel final : public user_op::OpKernel {
 public:
  EmbeddingPrefetchKernel() = default;
  ~EmbeddingPrefetchKernel() = default;

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

    const user_op::Tensor* num_unique_ids = ctx->Tensor4ArgNameAndIndex("num_unique_ids", 0);
    const user_op::Tensor* unique_ids = ctx->Tensor4ArgNameAndIndex("unique_ids", 0);
    const user_op::Tensor* column_ids = ctx->Tensor4ArgNameAndIndex("column_ids", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const int64_t embedding_size = ctx->Attr<int64_t>("embedding_size");
    const int64_t line_size = ctx->Attr<int64_t>("line_size");
    uint32_t num_unique;
    T* values_ptr = nullptr;
    LookupAndInitMissing<T, U, IDX>(ctx->stream(), kernel_state, unique_ids->shape().elem_cnt(),
                                    embedding_size, line_size, num_unique_ids->dptr(),
                                    unique_ids->dptr(), column_ids->dptr(), values_ptr,
                                    tmp_buffer->mut_dptr(), &num_unique, true);
    CHECK_GT(num_unique, 0);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define EMBEDDING_DATA_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(float, DataType::kFloat)

#define ID_DATA_TYPE_SEQ                            \
  OF_PP_MAKE_TUPLE_SEQ(uint32_t, DataType::kUInt32) \
  OF_PP_MAKE_TUPLE_SEQ(uint64_t, DataType::kUInt64) \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32)   \
  OF_PP_MAKE_TUPLE_SEQ(int64_t, DataType::kInt64)

#define IDX_DATA_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(uint32_t, DataType::kUInt32)

#define REGISTER_CUDA_EMBEDDING_PREFETCH_KERNEL(t_dtype_pair, column_dtype_pair, idx_dtype_pair) \
  REGISTER_USER_KERNEL("embedding_prefetch")                                                     \
      .SetCreateFn<EmbeddingPrefetchKernel<OF_PP_PAIR_FIRST(t_dtype_pair),                       \
                                           OF_PP_PAIR_FIRST(column_dtype_pair),                  \
                                           OF_PP_PAIR_FIRST(idx_dtype_pair)>>()                  \
      .SetIsMatchedHob(                                                                          \
          (user_op::HobDeviceType() == DeviceType::kCUDA)                                        \
          && (user_op::HobDataType("column_ids", 0) == OF_PP_PAIR_SECOND(column_dtype_pair))     \
          && (user_op::HobDataType("num_unique_ids", 0) == OF_PP_PAIR_SECOND(idx_dtype_pair)))   \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                        \
        const user_op::TensorDesc& unique_ids = ctx->InputTensorDesc("unique_ids", 0);           \
        EmbeddingTmpBufferManager buffer_manager(                                                \
            nullptr, unique_ids.shape().elem_cnt(),                                              \
            ctx->Attr<int64_t>("line_size") * sizeof(OF_PP_PAIR_FIRST(t_dtype_pair)), true);     \
        return buffer_manager.TotalBufferSize();                                                 \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_CUDA_EMBEDDING_PREFETCH_KERNEL, EMBEDDING_DATA_TYPE_SEQ,
                                 ID_DATA_TYPE_SEQ, IDX_DATA_TYPE_SEQ)

template<typename T, typename U, typename IDX>
class EmbeddingLookupKernel final : public user_op::OpKernel {
 public:
  EmbeddingLookupKernel() = default;
  ~EmbeddingLookupKernel() = default;

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
    const user_op::Tensor* num_unique_ids = ctx->Tensor4ArgNameAndIndex("num_unique_ids", 0);
    const user_op::Tensor* unique_ids = ctx->Tensor4ArgNameAndIndex("unique_ids", 0);
    const user_op::Tensor* column_ids = ctx->Tensor4ArgNameAndIndex("column_ids", 0);
    user_op::Tensor* unique_values = ctx->Tensor4ArgNameAndIndex("unique_values", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const int64_t embedding_size = ctx->Attr<int64_t>("embedding_size");
    const int64_t line_size = ctx->Attr<int64_t>("line_size");
    uint32_t num_unique;
    LookupAndInitMissing<T, U, IDX>(
        ctx->stream(), kernel_state, unique_ids->shape().elem_cnt(), embedding_size, line_size,
        num_unique_ids->dptr(), unique_ids->dptr(), column_ids->dptr(),
        unique_values->mut_dptr<T>(), tmp_buffer->mut_dptr(), &num_unique, false);
    CHECK_GT(num_unique, 0);
    if (ctx->has_output("embeddings", 0)) {
      user_op::Tensor* embeddings = ctx->Tensor4ArgNameAndIndex("embeddings", 0);
      CopyValuesToEmbeddings<T>(ctx->stream(), num_unique, embedding_size, line_size,
                                unique_values->data_type(), embeddings->data_type(),
                                unique_values->dptr<T>(), embeddings->mut_dptr());
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_EMBEDDING_LOOKUP_KERNEL(t_dtype_pair, column_dtype_pair, idx_dtype_pair) \
  REGISTER_USER_KERNEL("embedding_lookup")                                                     \
      .SetCreateFn<EmbeddingLookupKernel<OF_PP_PAIR_FIRST(t_dtype_pair),                       \
                                         OF_PP_PAIR_FIRST(column_dtype_pair),                  \
                                         OF_PP_PAIR_FIRST(idx_dtype_pair)>>()                  \
      .SetIsMatchedHob(                                                                        \
          (user_op::HobDeviceType() == DeviceType::kCUDA)                                      \
          && (user_op::HobDataType("unique_values", 0) == OF_PP_PAIR_SECOND(t_dtype_pair))     \
          && (user_op::HobDataType("column_ids", 0) == OF_PP_PAIR_SECOND(column_dtype_pair))   \
          && (user_op::HobDataType("num_unique_ids", 0) == OF_PP_PAIR_SECOND(idx_dtype_pair))) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                      \
        const user_op::TensorDesc& unique_ids = ctx->InputTensorDesc("unique_ids", 0);         \
        EmbeddingTmpBufferManager buffer_manager(                                              \
            nullptr, unique_ids.shape().elem_cnt(),                                            \
            ctx->Attr<int64_t>("line_size") * sizeof(OF_PP_PAIR_FIRST(t_dtype_pair)), false);  \
        return buffer_manager.TotalBufferSize();                                               \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_CUDA_EMBEDDING_LOOKUP_KERNEL, EMBEDDING_DATA_TYPE_SEQ,
                                 ID_DATA_TYPE_SEQ, IDX_DATA_TYPE_SEQ)

template<typename IDX>
class EmbeddingPutKernel final : public user_op::OpKernel {
 public:
  EmbeddingPutKernel() = default;
  ~EmbeddingPutKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<EmbeddingPutKernelState<IDX>>(ctx);
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    auto* kernel_state = dynamic_cast<EmbeddingPutKernelState<IDX>*>(state);
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

#define REGISTER_CUDA_EMBEDDING_PUT_KERNEL(dtype, typeproto)           \
  REGISTER_USER_KERNEL("embedding_put")                                \
      .SetCreateFn<EmbeddingPutKernel<dtype>>()                        \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("num_unique_ids", 0) == typeproto));

OF_PP_FOR_EACH_TUPLE(REGISTER_CUDA_EMBEDDING_PUT_KERNEL, IDX_DATA_TYPE_SEQ)

}  // namespace oneflow
