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
#include "oneflow/core/device/nccl_util.h"
#include "oneflow/core/job/eager_nccl_comm_manager.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/user/kernels/gather_kernel_util.h"
#include "oneflow/user/kernels/unsorted_segment_sum_kernel_util.h"
#include "oneflow/core/cuda/atomic.cuh"
#include "oneflow/core/embedding/hash_functions.cuh"
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/ep/include/primitive/copy_nd.h"
#include "oneflow/core/cuda/atomic.cuh"
#include "oneflow/core/embedding/embedding_manager.h"
#include "oneflow/user/kernels/one_embedding_data_shuffle.cuh"

namespace oneflow {

namespace {

enum class IdShuffleBufferType {
  kNumPartitionedUnique = 0,
  kPartitionedUniqueIds,
  kReceivedIds,
  kTableIds,
  kPartitionedUniqueTableIds,
  kReceivedTableIds,
  kWorkspace,
  kMaxType
};

template<typename K, typename U, typename IDX>
class IdShuffleTmpBufferManager final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IdShuffleTmpBufferManager);
  IdShuffleTmpBufferManager(void* ptr, const int64_t num_ids, const int64_t parallel_num,
                            bool need_table_ids, bool need_process_table_ids)
      : offset_(0),
        offsets_(static_cast<size_t>(IdShuffleBufferType::kMaxType), -1),
        sizes_(static_cast<size_t>(IdShuffleBufferType::kMaxType)),
        ptr_(ptr) {
    const int64_t num_table_ids = need_process_table_ids ? num_ids : 0;
    const size_t table_ids_bytes = need_table_ids ? num_ids * sizeof(U) : 0;
    AllocBuffer(IdShuffleBufferType::kNumPartitionedUnique, parallel_num * sizeof(IDX));
    size_t partitioned_ids_bytes = parallel_num * num_ids * sizeof(K);
    AllocBuffer(IdShuffleBufferType::kPartitionedUniqueIds, partitioned_ids_bytes);
    AllocBuffer(IdShuffleBufferType::kReceivedIds, partitioned_ids_bytes);
    AllocBuffer(IdShuffleBufferType::kTableIds, table_ids_bytes);
    size_t partitioned_table_ids_bytes = parallel_num * num_table_ids * sizeof(U);
    AllocBuffer(IdShuffleBufferType::kPartitionedUniqueTableIds, partitioned_table_ids_bytes);
    AllocBuffer(IdShuffleBufferType::kReceivedTableIds, partitioned_table_ids_bytes);
    const size_t hash_table_capacity = parallel_num * num_ids;
    AllocBuffer(IdShuffleBufferType::kWorkspace,
                hash_table_capacity * sizeof(data_shuffle::TableEntry<K>));
  }

  template<typename T = void>
  T* Ptr(IdShuffleBufferType type) {
    CHECK(ptr_ != nullptr);
    int64_t offset = offsets_.at(static_cast<size_t>(type));
    CHECK_NE(offset, -1);
    return reinterpret_cast<T*>(reinterpret_cast<char*>(ptr_) + offset);
  }

  int64_t Size(IdShuffleBufferType type) { return sizes_.at(static_cast<size_t>(type)); }

  size_t TotalBufferSize() const { return offset_; }

 private:
  void AllocBuffer(IdShuffleBufferType type, size_t size) {
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

template<typename IDX>
class DataShuffleKernelState final : public user_op::OpKernelState {
 public:
  explicit DataShuffleKernelState(user_op::KernelInitContext* ctx)
      : device_index_(-1),
        stream_name_(EagerNcclCommMgr::kDefaultStreamName),
        parallel_desc_(ctx->parallel_desc()) {
    OF_CUDA_CHECK(cudaGetDevice(&device_index_));
    if (ctx->op_conf().has_stream_name_hint()) { stream_name_ = ctx->op_conf().stream_name_hint(); }
    OF_CUDA_CHECK(cudaMallocHost(&host_num_keys_, sizeof(IDX)));
    OF_CUDA_CHECK(cudaMallocHost(
        &host_num_unique_matrix_,
        parallel_desc_.parallel_num() * parallel_desc_.parallel_num() * sizeof(IDX)));
    const std::string& embedding_name = ctx->Attr<std::string>("embedding_name");
    const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
    embedding_state_ = Singleton<embedding::EmbeddingManager>::Get()->GetEmbeddingState(
        embedding_name, parallel_id);
  }
  ~DataShuffleKernelState() {
    CudaCurrentDeviceGuard guard(device_index_);
    OF_CUDA_CHECK(cudaFreeHost(host_num_unique_matrix_));
  }

  ncclComm_t comm() { return GetOrCreate().comm; }

  IDX* HostNumUniqueMatrix() { return host_num_unique_matrix_; }
  IDX* HostNumKeys() { return host_num_keys_; }

  embedding::EmbeddingState* EmbeddingState() { return embedding_state_; }

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
  bool has_independent_stream_;
  std::string stream_name_;
  ParallelDesc parallel_desc_;
  std::unique_ptr<Comm> comm_;
  IDX* host_num_unique_matrix_;
  IDX* host_num_keys_;
  embedding::EmbeddingState* embedding_state_;
};

}  // namespace

template<typename K, typename U, typename IDX>
class IdShuffleKernel final : public user_op::OpKernel {
 public:
  IdShuffleKernel() : current_iter_(0){};
  ~IdShuffleKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<DataShuffleKernelState<IDX>>(ctx);
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    auto* kernel_state = dynamic_cast<DataShuffleKernelState<IDX>*>(state);
    CHECK(kernel_state != nullptr);
    const user_op::Tensor* ids = ctx->Tensor4ArgNameAndIndex("ids", 0);
    user_op::Tensor* num_unique_matrix = ctx->Tensor4ArgNameAndIndex("num_unique_matrix", 0);
    user_op::Tensor* inverse_unique_partition_indices =
        ctx->Tensor4ArgNameAndIndex("inverse_unique_partition_indices", 0);
    user_op::Tensor* cur_rank_num_unique = ctx->Tensor4ArgNameAndIndex("cur_rank_num_unique", 0);
    user_op::Tensor* cur_rank_unique_ids = ctx->Tensor4ArgNameAndIndex("cur_rank_unique_ids", 0);
    user_op::Tensor* cur_rank_unique_table_ids =
        ctx->Tensor4ArgNameAndIndex("cur_rank_unique_table_ids", 0);
    user_op::Tensor* cur_rank_inverse_indices =
        ctx->Tensor4ArgNameAndIndex("cur_rank_inverse_indices", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const int32_t num_tables = ctx->Attr<int32_t>("num_tables");
    const int64_t padding_idx = ctx->Attr<int64_t>("padding_idx");
    const bool has_padding_idx = ctx->Attr<bool>("has_padding_idx");
    const bool has_table_ids = ctx->has_input("table_ids", 0);
    const bool need_gen_table_ids = (!has_table_ids && num_tables > 1);
    const bool need_process_table_ids = (has_table_ids || num_tables > 1);
    const int64_t num_ids = ids->shape_view().elem_cnt();
    const int64_t parallel_num = ctx->parallel_ctx().parallel_num();
    const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
    cudaStream_t cuda_stream = ctx->stream()->As<ep::CudaStream>()->cuda_stream();
    IdShuffleTmpBufferManager<K, U, IDX> buffer_manager(
        tmp_buffer->mut_dptr(), num_ids, parallel_num, need_gen_table_ids, need_process_table_ids);
    CHECK_GE(tmp_buffer->shape_view().elem_cnt(), buffer_manager.TotalBufferSize());

    ncclComm_t comm = kernel_state->comm();
    IDX* host_num_unique_matrix = kernel_state->HostNumUniqueMatrix();
    IDX* host_num_keys = kernel_state->HostNumKeys();
    data_shuffle::IdShuffleDataPtrs<K, U, IDX> data_ptrs;
    data_ptrs.ids_ptr = reinterpret_cast<const K*>(ids->dptr());
    if (has_table_ids) {
      const user_op::Tensor* table_ids = ctx->Tensor4ArgNameAndIndex("table_ids", 0);
      data_ptrs.table_ids_ptr = reinterpret_cast<const U*>(table_ids->dptr());
    } else if (need_gen_table_ids) {
      data_shuffle::GenerateTableIds<<<BlocksNum4ThreadsNum(num_ids), kCudaThreadsNumPerBlock, 0,
                                       cuda_stream>>>(
          num_ids, num_tables, buffer_manager.template Ptr<U>(IdShuffleBufferType::kTableIds));
      data_ptrs.table_ids_ptr = buffer_manager.template Ptr<U>(IdShuffleBufferType::kTableIds);
    } else {
      data_ptrs.table_ids_ptr = nullptr;
    }
    data_ptrs.num_partitioned_unique =
        buffer_manager.template Ptr<IDX>(IdShuffleBufferType::kNumPartitionedUnique);
    data_ptrs.partitioned_unique_ids =
        buffer_manager.template Ptr<K>(IdShuffleBufferType::kPartitionedUniqueIds);
    data_ptrs.partitioned_unique_table_ids =
        buffer_manager.template Ptr<U>(IdShuffleBufferType::kPartitionedUniqueTableIds);
    data_ptrs.workspace_ptr = buffer_manager.Ptr(IdShuffleBufferType::kWorkspace);
    data_ptrs.workspace_size = buffer_manager.Size(IdShuffleBufferType::kWorkspace);
    data_ptrs.received_ids = buffer_manager.template Ptr<K>(IdShuffleBufferType::kReceivedIds);
    data_ptrs.received_table_ids =
        buffer_manager.template Ptr<U>(IdShuffleBufferType::kReceivedTableIds);
    data_ptrs.num_unique_matrix_ptr = reinterpret_cast<IDX*>(num_unique_matrix->mut_dptr());
    data_ptrs.inverse_unique_partition_indices_ptr =
        reinterpret_cast<IDX*>(inverse_unique_partition_indices->mut_dptr());
    data_ptrs.cur_rank_num_unique_ptr = reinterpret_cast<IDX*>(cur_rank_num_unique->mut_dptr());
    data_ptrs.cur_rank_unique_ids_ptr = reinterpret_cast<K*>(cur_rank_unique_ids->mut_dptr());
    data_ptrs.cur_rank_unique_table_ids_ptr =
        reinterpret_cast<U*>(cur_rank_unique_table_ids->mut_dptr());
    data_ptrs.cur_rank_inverse_indices_ptr =
        reinterpret_cast<IDX*>(cur_rank_inverse_indices->mut_dptr());

    data_shuffle::IdShuffle(ctx->stream(), comm, data_ptrs, num_ids, parallel_id, parallel_num,
                            num_unique_matrix->data_type(), ids->data_type(),
                            cur_rank_unique_table_ids->data_type(), need_process_table_ids,
                            has_padding_idx, padding_idx, host_num_unique_matrix, host_num_keys);

    embedding::EmbeddingState* embedding_state = kernel_state->EmbeddingState();
    std::vector<uint32_t> num_unique_matrix_vec(parallel_num * parallel_num);
    std::memcpy(num_unique_matrix_vec.data(), host_num_unique_matrix,
                parallel_num * parallel_num * sizeof(IDX));
    CHECK_EQ(sizeof(IDX), sizeof(uint32_t)) << "assume sizeof(IDX) equals to sizeof(uint32_t)";
    embedding_state->SetIdNumUniqueMatrix(num_unique_matrix_vec, current_iter_);

    uint32_t final_num_unique = *host_num_keys;
    embedding_state->SetIdFinalNumUnique(final_num_unique, current_iter_);
    current_iter_++;
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  mutable int64_t current_iter_;
};

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

#define REGISTER_CUDA_ID_SHUFFLE_KERNEL(k_dtype_pair, table_id_dtype_pair, idx_dtype_pair)       \
  REGISTER_USER_KERNEL("id_shuffle")                                                             \
      .SetCreateFn<                                                                              \
          IdShuffleKernel<OF_PP_PAIR_FIRST(k_dtype_pair), OF_PP_PAIR_FIRST(table_id_dtype_pair), \
                          OF_PP_PAIR_FIRST(idx_dtype_pair)>>()                                   \
      .SetIsMatchedHob(                                                                          \
          (user_op::HobDeviceType() == DeviceType::kCUDA)                                        \
          && (user_op::HobDataType("ids", 0) == OF_PP_PAIR_SECOND(k_dtype_pair))                 \
          && (user_op::HobDataType("cur_rank_unique_table_ids", 0)                               \
              == OF_PP_PAIR_SECOND(table_id_dtype_pair))                                         \
          && (user_op::HobDataType("num_unique_matrix", 0) == OF_PP_PAIR_SECOND(idx_dtype_pair)) \
          && (!ParseBooleanFromEnv("ONEFLOW_ONE_EMBEDDING_ID_SHUFFLE_USE_P2P", false)))          \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                        \
        const user_op::TensorDesc& ids = ctx->InputTensorDesc("ids", 0);                         \
        const bool has_table_ids = ctx->has_input("table_ids", 0);                               \
        const int32_t num_tables = ctx->Attr<int32_t>("num_tables");                             \
        const bool need_gen_table_ids = (!has_table_ids && num_tables > 1);                      \
        const bool need_process_table_ids = (has_table_ids || num_tables > 1);                   \
        IdShuffleTmpBufferManager<OF_PP_PAIR_FIRST(k_dtype_pair),                                \
                                  OF_PP_PAIR_FIRST(table_id_dtype_pair),                         \
                                  OF_PP_PAIR_FIRST(idx_dtype_pair)>                              \
            buffer_manager(nullptr, ids.shape().elem_cnt(), ctx->parallel_desc().parallel_num(), \
                           need_gen_table_ids, need_process_table_ids);                          \
        return buffer_manager.TotalBufferSize();                                                 \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_CUDA_ID_SHUFFLE_KERNEL, ID_DATA_TYPE_SEQ,
                                 TABLE_ID_DATA_TYPE_SEQ, IDX_DATA_TYPE_SEQ)

__device__ float RoundHalfAwayFromZero(const float x) {
  float abs_val = abs(x);
  float floor_val = floor(abs_val + static_cast<float>(0.5));
  return copysignf(floor_val, x);
}

// warp reduce version.
constexpr int32_t kWarpSize = 32;
constexpr int32_t kMaxColSize = 1024;

template<typename T, int thread_group_width = kWarpSize>
__inline__ __device__ T WarpMaxAllReduce(T val) {
  for (int32_t lane_mask = thread_group_width / 2; lane_mask > 0; lane_mask /= 2) {
    val = max(val, __shfl_xor_sync(0xffffffff, val, lane_mask, thread_group_width));
  }
  return val;
}

inline cudaError_t GetWarpImplNumBlocks(int64_t block_size, int64_t max_blocks, int64_t waves,
                                        int* num_blocks) {
  int dev;
  {
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) { return err; }
  }
  int sm_count;
  {
    cudaError_t err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) { return err; }
  }
  int tpm;
  {
    cudaError_t err = cudaDeviceGetAttribute(&tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
    if (err != cudaSuccess) { return err; }
  }
  *num_blocks =
      std::max<int>(1, std::min<int64_t>(max_blocks, sm_count * tpm / block_size * waves));
  return cudaSuccess;
}

template<typename T, typename ComputeType, int pack_size, int cols_per_thread,
         int thread_group_width, int rows_per_access, bool padding>
__global__ void QuantizeWarpImplKernel(const T* src, int8_t* dst, T* quantize_factor,
                                       const int64_t rows, const int64_t cols) {
  static_assert(cols_per_thread % pack_size == 0, "");
  static_assert(thread_group_width <= kWarpSize, "");
  static_assert(kWarpSize % thread_group_width == 0, "");
  constexpr int num_packs = cols_per_thread / pack_size;
  assert(cols <= cols_per_thread * thread_group_width);
  ComputeType buf[rows_per_access][cols_per_thread];
  const int global_thread_group_id = blockIdx.x * blockDim.y + threadIdx.y;
  const int num_global_thread_group = gridDim.x * blockDim.y;
  const int lane_id = threadIdx.x;
  const int64_t step = num_global_thread_group * rows_per_access;
  using LoadType = cuda::elementwise::PackType<T, pack_size>;
  using LoadPack = cuda::elementwise::Pack<T, pack_size>;
  using StoreType = cuda::elementwise::PackType<int8_t, pack_size>;
  using StorePack = cuda::elementwise::Pack<int8_t, pack_size>;

  for (int64_t row = global_thread_group_id * rows_per_access; row < rows; row += step) {
    ComputeType thread_abs_max[rows_per_access];
#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; row_id++) {
      ComputeType* row_buf = buf[row_id];
      thread_abs_max[row_id] = 0.0;
#pragma unroll
      for (int pack_id = 0; pack_id < num_packs; pack_id++) {
        const int pack_offset = pack_id * pack_size;
        const int col = (pack_id * thread_group_width + lane_id) * pack_size;
        LoadPack load_pack;
        if (!padding || col < cols) {
          const int64_t load_offset = ((row + row_id) * cols + col) / pack_size;
          load_pack.storage = *(reinterpret_cast<const LoadType*>(src) + load_offset);
#pragma unroll
          for (int i = 0; i < pack_size; i++) {
            row_buf[pack_offset + i] = static_cast<ComputeType>(load_pack.elem[i]);
            thread_abs_max[row_id] = max(thread_abs_max[row_id], abs(row_buf[pack_offset + i]));
          }
        } else {
#pragma unroll
          for (int i = 0; i < pack_size; i++) { row_buf[pack_offset + i] = 0.0; }
        }
      }
    }
    ComputeType warp_max[rows_per_access];
#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; row_id++) {
      warp_max[row_id] = WarpMaxAllReduce<ComputeType, thread_group_width>(thread_abs_max[row_id]);
      if (threadIdx.x == 0) { quantize_factor[row + row_id] = static_cast<T>(warp_max[row_id]); }
      ComputeType* row_buf = buf[row_id];
      ComputeType quantize_factor_val = static_cast<ComputeType>(127.0) / warp_max[row_id];
#pragma unroll
      for (int col = 0; col < cols_per_thread; col++) {
        row_buf[col] = RoundHalfAwayFromZero(row_buf[col] * quantize_factor_val);
      }
#pragma unroll
      for (int pack_id = 0; pack_id < num_packs; pack_id++) {
        const int pack_offset = pack_id * pack_size;
        const int col = (pack_id * thread_group_width + lane_id) * pack_size;
        StorePack store_pack;
        if (!padding || col < cols) {
          const int64_t store_offset = ((row + row_id) * cols + col) / pack_size;
          for (int i = 0; i < pack_size; i++) {
            store_pack.elem[i] = static_cast<int8_t>(row_buf[pack_id * pack_size + i]);
          }
          *(reinterpret_cast<StoreType*>(dst) + store_offset) = store_pack.storage;
        }
      }
    }
  }
}

template<typename T, typename ComputeType, int pack_size, int cols_per_thread,
         int thread_group_width, int rows_per_access, bool padding>
inline cudaError_t LaunchQuantizeWarpImpl(cudaStream_t stream, const T* src, int8_t* dst,
                                          T* quantize_factor, const int64_t rows,
                                          const int64_t cols) {
  constexpr int block_size = 128;
  constexpr int waves = 32;
  static_assert(block_size % thread_group_width == 0, "");
  constexpr int thread_groups_per_block = block_size / thread_group_width;
  dim3 block_dim(thread_group_width, thread_groups_per_block);
  const int64_t num_blocks =
      (rows / rows_per_access + thread_groups_per_block - 1) / thread_groups_per_block;
  int grid_dim_x = 0;

  cudaError_t err = GetWarpImplNumBlocks(block_size, num_blocks, waves, &grid_dim_x);
  if (err != cudaSuccess) { return err; }

  QuantizeWarpImplKernel<T, ComputeType, pack_size, cols_per_thread, thread_group_width,
                         rows_per_access, padding>
      <<<grid_dim_x, block_dim, 0, stream>>>(src, dst, quantize_factor, rows, cols);
  return cudaPeekAtLastError();
}

template<typename T, typename ComputeType, int pack_size, int cols_per_thread,
         int thread_group_width, int rows_per_access>
inline cudaError_t DispatchQuantizeWarpImplPadding(cudaStream_t stream, const T* src, int8_t* dst,
                                                   T* quantize_factor, const int64_t rows,
                                                   const int64_t cols) {
  if (cols == cols_per_thread * thread_group_width) {
    return LaunchQuantizeWarpImpl<T, ComputeType, pack_size, cols_per_thread, thread_group_width,
                                  rows_per_access, false>(stream, src, dst, quantize_factor, rows,
                                                          cols);
  } else {
    return LaunchQuantizeWarpImpl<T, ComputeType, pack_size, cols_per_thread, thread_group_width,
                                  rows_per_access, true>(stream, src, dst, quantize_factor, rows,
                                                         cols);
  }
}

template<typename T, typename ComputeType, int pack_size>
typename std::enable_if<pack_size == 1, cudaError_t>::type DispatchQuantizeWarpImplCols(
    cudaStream_t stream, const T* src, int8_t* dst, T* quantize_factor, const int64_t rows,
    const int64_t cols) {
  if (cols <= 0) { return cudaErrorInvalidValue; }
#define DEFINE_ONE_ELIF(thread_group_width)                                                       \
  else if (cols <= (thread_group_width)*pack_size) {                                              \
    if (rows % 2 == 0) {                                                                          \
      return DispatchQuantizeWarpImplPadding<T, ComputeType, pack_size, pack_size,                \
                                             thread_group_width, 2>(stream, src, dst,             \
                                                                    quantize_factor, rows, cols); \
    } else {                                                                                      \
      return DispatchQuantizeWarpImplPadding<T, ComputeType, pack_size, pack_size,                \
                                             thread_group_width, 1>(stream, src, dst,             \
                                                                    quantize_factor, rows, cols); \
    }                                                                                             \
  }
  DEFINE_ONE_ELIF(1)
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(col)                                                              \
  else if (cols <= (col)*kWarpSize) {                                                     \
    return DispatchQuantizeWarpImplPadding<T, ComputeType, pack_size, col, kWarpSize, 1>( \
        stream, src, dst, quantize_factor, rows, cols);                                   \
  }
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(3)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(5)
  DEFINE_ONE_ELIF(6)
  DEFINE_ONE_ELIF(7)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(9)
  DEFINE_ONE_ELIF(10)
  DEFINE_ONE_ELIF(11)
  DEFINE_ONE_ELIF(12)
  DEFINE_ONE_ELIF(13)
  DEFINE_ONE_ELIF(14)
  DEFINE_ONE_ELIF(15)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(17)
  DEFINE_ONE_ELIF(18)
  DEFINE_ONE_ELIF(19)
  DEFINE_ONE_ELIF(20)
  DEFINE_ONE_ELIF(21)
  DEFINE_ONE_ELIF(22)
  DEFINE_ONE_ELIF(23)
  DEFINE_ONE_ELIF(24)
  DEFINE_ONE_ELIF(25)
  DEFINE_ONE_ELIF(26)
  DEFINE_ONE_ELIF(27)
  DEFINE_ONE_ELIF(28)
  DEFINE_ONE_ELIF(29)
  DEFINE_ONE_ELIF(30)
  DEFINE_ONE_ELIF(31)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
  else {
    return cudaErrorInvalidValue;
  }
}

template<typename T, typename ComputeType, int pack_size>
typename std::enable_if<pack_size == 2, cudaError_t>::type DispatchQuantizeWarpImplCols(
    cudaStream_t stream, const T* src, int8_t* dst, T* quantize_factor, const int64_t rows,
    const int64_t cols) {
  if (cols <= 0) { return cudaErrorInvalidValue; }
#define DEFINE_ONE_ELIF(thread_group_width)                                                       \
  else if (cols <= (thread_group_width)*pack_size) {                                              \
    if (rows % 2 == 0) {                                                                          \
      return DispatchQuantizeWarpImplPadding<T, ComputeType, pack_size, pack_size,                \
                                             thread_group_width, 2>(stream, src, dst,             \
                                                                    quantize_factor, rows, cols); \
    } else {                                                                                      \
      return DispatchQuantizeWarpImplPadding<T, ComputeType, pack_size, pack_size,                \
                                             thread_group_width, 1>(stream, src, dst,             \
                                                                    quantize_factor, rows, cols); \
    }                                                                                             \
  }
  DEFINE_ONE_ELIF(1)
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(col)                                                              \
  else if (cols <= (col)*kWarpSize) {                                                     \
    return DispatchQuantizeWarpImplPadding<T, ComputeType, pack_size, col, kWarpSize, 1>( \
        stream, src, dst, quantize_factor, rows, cols);                                   \
  }
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(6)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(10)
  DEFINE_ONE_ELIF(12)
  DEFINE_ONE_ELIF(14)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(18)
  DEFINE_ONE_ELIF(20)
  DEFINE_ONE_ELIF(22)
  DEFINE_ONE_ELIF(24)
  DEFINE_ONE_ELIF(26)
  DEFINE_ONE_ELIF(28)
  DEFINE_ONE_ELIF(30)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
  else {
    return cudaErrorInvalidValue;
  }
}

template<typename T, typename ComputeType>
struct DispatchQuantizeWarpImplPackSize {
  cudaError_t operator()(cudaStream_t stream, const T* src, int8_t* dst, T* quantize_factor,
                         const int64_t rows, const int64_t cols) {
    if (cols % 2 == 0) {
      return DispatchQuantizeWarpImplCols<T, ComputeType, 2>(stream, src, dst, quantize_factor,
                                                             rows, cols);
    } else {
      return DispatchQuantizeWarpImplCols<T, ComputeType, 1>(stream, src, dst, quantize_factor,
                                                             rows, cols);
    }
  }
};

template<typename T, typename ComputeType, typename IDX, int pack_size>
__global__ void DequantizeKernel(const int8_t* x, T* quantize_factor, T* out, IDX col_size,
                                 IDX elem_cnt);

template<typename T, typename ComputeType, typename IDX, int pack_size>
__global__ void DequantizeKernel(const int8_t* x, T* quantize_factor, T* out, IDX col_size,
                                 IDX elem_cnt) {
  IDX global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  for (int index = global_thread_id * pack_size; index < elem_cnt;
       index += gridDim.x * blockDim.x * pack_size) {
    IDX quantize_factor_idx = index / col_size;
    ComputeType quantize_factor_val = static_cast<ComputeType>(quantize_factor[quantize_factor_idx])
                                      / static_cast<ComputeType>(127.0);
    using LoadPackType = cuda::elementwise::PackType<int8_t, pack_size>;
    using LoadPack = cuda::elementwise::Pack<int8_t, pack_size>;
    using StorePackType = cuda::elementwise::PackType<T, pack_size>;
    using StorePack = cuda::elementwise::Pack<T, pack_size>;
    LoadPack load_pack{};
    StorePack store_pack{};
    load_pack.storage = *(reinterpret_cast<const LoadPackType*>(x) + index / pack_size);
#pragma unroll
    for (int i = 0; i < pack_size; i++) {
      store_pack.elem[i] =
          static_cast<T>(static_cast<ComputeType>(load_pack.elem[i]) * quantize_factor_val);
    }
    *(reinterpret_cast<StorePackType*>(out) + index / pack_size) = store_pack.storage;
  }
}

template<typename T, typename ComputeType, typename IDX, int pack_size>
cudaError_t DispatchDequantizeKernelPackSize(cudaStream_t stream, const int8_t* src,
                                             T* quantize_factor, T* dst, const int64_t col_size,
                                             const int64_t elem_cnt) {
  const int64_t pack_num = elem_cnt / pack_size;
  int grid_size = 0;
  cudaError_t err = cuda::elementwise::GetNumBlocks(pack_num, &grid_size);
  if (err != cudaSuccess) { return err; }
  DequantizeKernel<T, ComputeType, IDX, pack_size>
      <<<grid_size, cuda::elementwise::kBlockSize, 0, stream>>>(src, quantize_factor, dst, col_size,
                                                                elem_cnt);
  return cudaSuccess;
}

template<typename T, typename ComputeType, typename IDX>
inline cudaError_t LaunchDequantizeKernel(cudaStream_t stream, const int8_t* src,
                                          T* quantize_factor, T* dst, const int64_t col_size,
                                          const int64_t elem_cnt) {
  constexpr int quantized_src_pack_size = cuda::elementwise::PackSize<int8_t>();
  constexpr int dst_pack_size = cuda::elementwise::PackSize<T>();
  int launch_pack_size = std::min(quantized_src_pack_size, dst_pack_size);
  if (launch_pack_size == 8 && col_size % 8 == 0) {
    cudaError_t err = DispatchDequantizeKernelPackSize<T, ComputeType, IDX, 8>(
        stream, src, quantize_factor, dst, col_size, elem_cnt);
    if (err != cudaSuccess) { return err; }
  } else if (launch_pack_size == 4 && col_size % 4 == 0) {
    cudaError_t err = DispatchDequantizeKernelPackSize<T, ComputeType, IDX, 4>(
        stream, src, quantize_factor, dst, col_size, elem_cnt);
    if (err != cudaSuccess) { return err; }
  } else if (launch_pack_size == 2 && col_size % 2 == 0) {
    cudaError_t err = DispatchDequantizeKernelPackSize<T, ComputeType, IDX, 2>(
        stream, src, quantize_factor, dst, col_size, elem_cnt);
    if (err != cudaSuccess) { return err; }
  } else {
    cudaError_t err = DispatchDequantizeKernelPackSize<T, ComputeType, IDX, 1>(
        stream, src, quantize_factor, dst, col_size, elem_cnt);
    if (err != cudaSuccess) { return err; }
  }
  return cudaPeekAtLastError();
}

template<typename T>
struct DefaultComputeType {
  using type = T;
};

template<>
struct DefaultComputeType<half> {
  using type = float;
};

template<typename T, typename IDX>
class EmbeddingShuffleKernel final : public user_op::OpKernel {
 public:
  EmbeddingShuffleKernel() : current_iter_(0) {}
  ~EmbeddingShuffleKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<DataShuffleKernelState<IDX>>(ctx);
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    auto* kernel_state = dynamic_cast<DataShuffleKernelState<IDX>*>(state);
    CHECK(kernel_state != nullptr);
    embedding::EmbeddingState* embedding_state = kernel_state->EmbeddingState();
    std::unique_ptr<embedding::TmpBufferAllocator> allocator =
        embedding_state->NewTmpBufferAllocator(ctx);
    embedding_state->OnEmbeddingShuffleStart(ctx, current_iter_);
    const user_op::Tensor* num_unique_matrix = ctx->Tensor4ArgNameAndIndex("num_unique_matrix", 0);
    const user_op::Tensor* cur_rank_inverse_indices =
        ctx->Tensor4ArgNameAndIndex("cur_rank_inverse_indices", 0);
    const user_op::Tensor* inverse_unique_partition_indices =
        ctx->Tensor4ArgNameAndIndex("inverse_unique_partition_indices", 0);
    user_op::Tensor* embeddings = ctx->Tensor4ArgNameAndIndex("embeddings", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    ncclComm_t comm = kernel_state->comm();
    using ComputeType = typename DefaultComputeType<T>::type;
    const int64_t embedding_size = ctx->Attr<int64_t>("embedding_size");
    IDX* host_num_unique_matrix = kernel_state->HostNumUniqueMatrix();
    DataType data_type = embeddings->data_type();
    const int64_t num_ids = inverse_unique_partition_indices->shape_view().elem_cnt();
    const int64_t parallel_num = ctx->parallel_ctx().parallel_num();
    const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
    const bool skip_last_gather = ctx->Attr<bool>("skip_last_gather");
    bool enable_quantized_comm_env_var =
        ParseBooleanFromEnv("ONEFLOW_ONE_EMBEDDING_ENABLE_QUANTIZED_COMM", false);
    bool enable_quantized_comm = enable_quantized_comm_env_var && (embedding_size < kMaxColSize);
    if (enable_quantized_comm_env_var && !enable_quantized_comm) {
      LOG(WARNING) << "Only envrionment variable ONEFLOW_ONE_EMBEDDING_ENABLE_QUANTIZED_COMM=1 and "
                      "embedding_size less equal than 1024 can use quantized communication. ";
    }
    cudaStream_t cuda_stream = ctx->stream()->As<ep::CudaStream>()->cuda_stream();
    const std::vector<uint32_t>& num_unique_matrix_vec =
        embedding_state->GetIdNumUniqueMatrix(current_iter_);
    CHECK_EQ(sizeof(IDX), sizeof(uint32_t)) << "assume sizeof(IDX) equals to sizeof(uint32_t)";
    ;
    std::memcpy(host_num_unique_matrix, num_unique_matrix_vec.data(),
                parallel_num * parallel_num * sizeof(IDX));
    uint32_t num_unique = embedding_state->GetIdNumUnique(current_iter_);

    int64_t cur_rank_num_ids = 0;
    for (int64_t i = 0; i < parallel_num; ++i) {
      cur_rank_num_ids += host_num_unique_matrix[i * parallel_num + parallel_id];
    }
    int64_t unique_partitioned_num_ids = 0;
    for (int64_t i = 0; i < parallel_num; ++i) {
      unique_partitioned_num_ids += host_num_unique_matrix[parallel_id * parallel_num + i];
    }
    const T* cur_rank_embeddings_ptr = reinterpret_cast<const T*>(
        embedding_state->EmbeddingShuffleCurRankEmbeddings(current_iter_));
    if (!enable_quantized_comm) {
      // 1. reverse cur_rank unique, from (num_unique, embedding_size) to (cur_rank_num_ids,
      // embedding_size)
      void* reverse_unique_cur_rank_embeddings;
      allocator->Allocate(&reverse_unique_cur_rank_embeddings,
                          cur_rank_num_ids * embedding_size * sizeof(T));
      GatherKernelUtilImpl<DeviceType::kCUDA, T, IDX>::Forward(
          ctx->stream(), reinterpret_cast<const IDX*>(cur_rank_inverse_indices->dptr()),
          cur_rank_num_ids, cur_rank_embeddings_ptr, Shape({1, num_unique, embedding_size}),
          reinterpret_cast<T*>(reverse_unique_cur_rank_embeddings), 0);

      // 2. send recv embedding, from (cur_rank_num_ids, embedding_size) to
      // (unique_partitioned_num_ids, embedding_size)
      if (skip_last_gather) {
        data_shuffle::ShuffleEmbeddings(cuda_stream, comm, parallel_id, parallel_num, num_ids,
                                        embedding_size, data_type, host_num_unique_matrix,
                                        reinterpret_cast<T*>(reverse_unique_cur_rank_embeddings),
                                        embeddings->mut_dptr<T>());
        allocator->Free(reverse_unique_cur_rank_embeddings);
      } else {
        void* received_embeddings;  // T
        allocator->Allocate(&received_embeddings, GetCudaAlignedSize(unique_partitioned_num_ids
                                                                     * embedding_size * sizeof(T)));

        data_shuffle::ShuffleEmbeddings(cuda_stream, comm, parallel_id, parallel_num, num_ids,
                                        embedding_size, data_type, host_num_unique_matrix,
                                        reinterpret_cast<T*>(reverse_unique_cur_rank_embeddings),
                                        reinterpret_cast<T*>(received_embeddings));
        allocator->Free(reverse_unique_cur_rank_embeddings);

        // 3. reverse unique_partition, from (unique_partitioned_num_ids, embedding_size) to
        // (num_ids, embedding_size)
        GatherKernelUtilImpl<DeviceType::kCUDA, T, IDX>::Forward(
            ctx->stream(), reinterpret_cast<const IDX*>(inverse_unique_partition_indices->dptr()),
            num_ids, reinterpret_cast<T*>(received_embeddings),
            Shape({1, unique_partitioned_num_ids, embedding_size}), embeddings->mut_dptr<T>(), 0);
        allocator->Free(received_embeddings);
      }
    } else {
      CHECK(!skip_last_gather) << "when enable_quantized_comm, should not use fuse kernel.";
      // 1. quantize cur_rank_embeddings, from (num_unique, embedding_size) T to (num_unique,
      // embedding_size) int8_t, and get (num_unique,) T factor
      void* quantize_cur_rank_embeddings;  // int8_t
      allocator->Allocate(&quantize_cur_rank_embeddings,
                          num_unique * embedding_size * sizeof(int8_t));
      void* cur_rank_quantize_factor;  // T
      allocator->Allocate(&cur_rank_quantize_factor, num_unique * sizeof(T));
      DispatchQuantizeWarpImplPackSize<T, ComputeType>()(
          cuda_stream, cur_rank_embeddings_ptr,
          reinterpret_cast<int8_t*>(quantize_cur_rank_embeddings),
          reinterpret_cast<T*>(cur_rank_quantize_factor), num_unique, embedding_size);
      // 2. reverse cur_rank unique, from (num_unique, embedding_size) to (cur_rank_num_ids,
      // embedding_size)
      void* reverse_unique_cur_rank_embeddings;  // int8_t

      allocator->Allocate(&reverse_unique_cur_rank_embeddings,
                          cur_rank_num_ids * embedding_size * sizeof(int8_t));

      GatherKernelUtilImpl<DeviceType::kCUDA, int8_t, IDX>::Forward(
          ctx->stream(), reinterpret_cast<const IDX*>(cur_rank_inverse_indices->dptr()),
          cur_rank_num_ids, reinterpret_cast<int8_t*>(quantize_cur_rank_embeddings),
          Shape({1, num_unique, embedding_size}),
          reinterpret_cast<int8_t*>(reverse_unique_cur_rank_embeddings), 0);
      allocator->Free(quantize_cur_rank_embeddings);

      // 3. reverse cur_rank quantize factor unique, from (num_unique) to (cur_rank_num_ids)
      void* reverse_cur_rank_quantize_factor;  // T
      allocator->Allocate(&reverse_cur_rank_quantize_factor, cur_rank_num_ids * sizeof(T));

      GatherKernelUtilImpl<DeviceType::kCUDA, T, IDX>::Forward(
          ctx->stream(), reinterpret_cast<const IDX*>(cur_rank_inverse_indices->dptr()),
          cur_rank_num_ids, reinterpret_cast<T*>(cur_rank_quantize_factor),
          Shape({1, num_unique, 1}), reinterpret_cast<T*>(reverse_cur_rank_quantize_factor), 0);
      allocator->Free(cur_rank_quantize_factor);
      // 4. send recv embedding and factor, from (cur_rank_num_ids, embedding_size) to
      // (unique_partitioned_num_ids, embedding_size)
      void* received_embeddings;   // int8_t
      void* recv_quantize_factor;  // T
      allocator->Allocate(&received_embeddings,
                          unique_partitioned_num_ids * embedding_size * sizeof(int8_t));
      allocator->Allocate(&recv_quantize_factor, unique_partitioned_num_ids * sizeof(T));

      data_shuffle::ShuffleEmbeddings(cuda_stream, comm, parallel_id, parallel_num, num_ids,
                                      embedding_size, data_type, host_num_unique_matrix,
                                      reinterpret_cast<int8_t*>(reverse_unique_cur_rank_embeddings),
                                      reinterpret_cast<int8_t*>(received_embeddings),
                                      reinterpret_cast<T*>(reverse_cur_rank_quantize_factor),
                                      reinterpret_cast<T*>(recv_quantize_factor));
      allocator->Free(reverse_unique_cur_rank_embeddings);
      allocator->Free(reverse_cur_rank_quantize_factor);

      // 5. reverse unique_partition, from (unique_partitioned_num_ids, embedding_size) to (num_ids,
      // embedding_size)
      void* reverse_recv_quantize_cur_rank_embeddings;  // int8_t
      allocator->Allocate(&reverse_recv_quantize_cur_rank_embeddings,
                          num_ids * embedding_size * sizeof(int8_t));

      GatherKernelUtilImpl<DeviceType::kCUDA, int8_t, IDX>::Forward(
          ctx->stream(), reinterpret_cast<const IDX*>(inverse_unique_partition_indices->dptr()),
          num_ids, reinterpret_cast<int8_t*>(received_embeddings),
          Shape({1, unique_partitioned_num_ids, embedding_size}),
          reinterpret_cast<int8_t*>(reverse_recv_quantize_cur_rank_embeddings), 0);
      allocator->Free(received_embeddings);
      // 6. reverse unique_partition_factor, from (unique_partitioned_num_ids) to (num_ids)
      void* reverse_recv_quantize_factor;  // T
      allocator->Allocate(&reverse_recv_quantize_factor, num_ids * sizeof(T));

      GatherKernelUtilImpl<DeviceType::kCUDA, T, IDX>::Forward(
          ctx->stream(), reinterpret_cast<const IDX*>(inverse_unique_partition_indices->dptr()),
          num_ids, reinterpret_cast<T*>(recv_quantize_factor),
          Shape({1, unique_partitioned_num_ids, 1}),
          reinterpret_cast<T*>(reverse_recv_quantize_factor), 0);
      allocator->Free(recv_quantize_factor);

      // 7. dequantize embeddings, from (num_ids, embedding_size) int8_t to (num_ids,
      // embedding_size) T
      int32_t dequantize_row_size = num_ids;
      IDX dequantize_elem_cnt = dequantize_row_size * embedding_size;
      OF_CUDA_CHECK((LaunchDequantizeKernel<T, ComputeType, IDX>(
          cuda_stream, reinterpret_cast<int8_t*>(reverse_recv_quantize_cur_rank_embeddings),
          reinterpret_cast<T*>(reverse_recv_quantize_factor), embeddings->mut_dptr<T>(),
          embedding_size, dequantize_elem_cnt)));
      allocator->Free(reverse_recv_quantize_cur_rank_embeddings);
      allocator->Free(reverse_recv_quantize_factor);
    }
    embedding_state->OnEmbeddingShuffleEnd(ctx, current_iter_);
    current_iter_++;
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  mutable int64_t current_iter_;
};

#define REGISTER_CUDA_EMBEDDING_SHUFFLE_KERNEL(t_dtype_pair, idx_dtype_pair)                      \
  REGISTER_USER_KERNEL("embedding_shuffle")                                                       \
      .SetCreateFn<EmbeddingShuffleKernel<OF_PP_PAIR_FIRST(t_dtype_pair),                         \
                                          OF_PP_PAIR_FIRST(idx_dtype_pair)>>()                    \
      .SetIsMatchedHob(                                                                           \
          (user_op::HobDeviceType() == DeviceType::kCUDA)                                         \
          && (user_op::HobDataType("cur_rank_embeddings", 0) == OF_PP_PAIR_SECOND(t_dtype_pair))  \
          && ((user_op::HobAttr<bool>("skip_last_gather") == false)                               \
              || (!embedding::UseEmbeddingShuffleP2PKernel(OF_PP_PAIR_SECOND(t_dtype_pair),       \
                                                           OF_PP_PAIR_SECOND(idx_dtype_pair))))   \
          && (user_op::HobDataType("num_unique_matrix", 0) == OF_PP_PAIR_SECOND(idx_dtype_pair))) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                         \
        const user_op::TensorDesc& inverse_unique_partition_indices =                             \
            ctx->InputTensorDesc("inverse_unique_partition_indices", 0);                          \
        const int64_t num_ids = inverse_unique_partition_indices.shape().elem_cnt();              \
        const int64_t parallel_num = ctx->parallel_ctx().parallel_num();                          \
        const int64_t cur_rank_max_num_ids = parallel_num * num_ids;                              \
        const int64_t embedding_size = ctx->Attr<int64_t>("embedding_size");                      \
        bool enable_quantized_comm =                                                              \
            ParseBooleanFromEnv("ONEFLOW_ONE_EMBEDDING_ENABLE_QUANTIZED_COMM", false)             \
            && (embedding_size < kMaxColSize);                                                    \
        size_t tmp_size = 0;                                                                      \
        if (embedding::UseDynamicMemoryAllocation()) { return tmp_size; }                         \
        if (!enable_quantized_comm) {                                                             \
          size_t reverse_cur_rank_embeddings_size = GetCudaAlignedSize(                           \
              cur_rank_max_num_ids * embedding_size * sizeof(OF_PP_PAIR_FIRST(t_dtype_pair)));    \
          size_t recv_unique_embeddings_size = reverse_cur_rank_embeddings_size;                  \
          tmp_size = reverse_cur_rank_embeddings_size + recv_unique_embeddings_size;              \
        } else {                                                                                  \
          size_t total_elem_cnt = cur_rank_max_num_ids * embedding_size;                          \
          size_t reverse_cur_rank_embeddings_size =                                               \
              GetCudaAlignedSize(total_elem_cnt * sizeof(int8_t));                                \
          size_t recv_unique_embeddings = reverse_cur_rank_embeddings_size;                       \
          size_t quantize_cur_rank_embeddings_size = reverse_cur_rank_embeddings_size;            \
          size_t reverse_recv_quantize_cur_rank_embeddings_size =                                 \
              reverse_cur_rank_embeddings_size;                                                   \
          size_t cur_rank_quantize_factor_size =                                                  \
              GetCudaAlignedSize(cur_rank_max_num_ids * sizeof(OF_PP_PAIR_FIRST(t_dtype_pair)));  \
          size_t reverse_cur_rank_quantize_factor_size = cur_rank_quantize_factor_size;           \
          size_t recv_quantize_factor_size = cur_rank_quantize_factor_size;                       \
          size_t reverse_recv_quantize_factor_size = cur_rank_quantize_factor_size;               \
          tmp_size = reverse_cur_rank_embeddings_size + recv_unique_embeddings                    \
                     + quantize_cur_rank_embeddings_size                                          \
                     + reverse_recv_quantize_cur_rank_embeddings_size                             \
                     + cur_rank_quantize_factor_size + reverse_cur_rank_quantize_factor_size      \
                     + recv_quantize_factor_size + reverse_recv_quantize_factor_size;             \
        }                                                                                         \
        return tmp_size;                                                                          \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_CUDA_EMBEDDING_SHUFFLE_KERNEL,
                                 FLOATING_DATA_TYPE_SEQ HALF_DATA_TYPE_SEQ, IDX_DATA_TYPE_SEQ)

template<typename T, typename IDX>
class EmbeddingGradientShuffleKernel final : public user_op::OpKernel {
 public:
  EmbeddingGradientShuffleKernel() : current_iter_(0){};
  ~EmbeddingGradientShuffleKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<DataShuffleKernelState<IDX>>(ctx);
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    auto* kernel_state = dynamic_cast<DataShuffleKernelState<IDX>*>(state);
    CHECK(kernel_state != nullptr);
    embedding::EmbeddingState* embedding_state = kernel_state->EmbeddingState();
    std::unique_ptr<embedding::TmpBufferAllocator> allocator =
        embedding_state->NewTmpBufferAllocator(ctx);
    const user_op::Tensor* embedding_grad = ctx->Tensor4ArgNameAndIndex("embedding_grad", 0);

    const user_op::Tensor* num_unique_matrix = ctx->Tensor4ArgNameAndIndex("num_unique_matrix", 0);
    const user_op::Tensor* cur_rank_inverse_indices =
        ctx->Tensor4ArgNameAndIndex("cur_rank_inverse_indices", 0);
    const user_op::Tensor* inverse_unique_partition_indices =
        ctx->Tensor4ArgNameAndIndex("inverse_unique_partition_indices", 0);
    user_op::Tensor* cur_rank_unique_embedding_grad =
        ctx->Tensor4ArgNameAndIndex("cur_rank_unique_embedding_grad", 0);
    const int64_t embedding_size = ctx->Attr<int64_t>("embedding_size");
    const bool only_zero_valid_grad = ctx->Attr<bool>("only_zero_valid_grad");
    IDX* host_num_unique_matrix = kernel_state->HostNumUniqueMatrix();
    DataType data_type = embedding_grad->data_type();
    const int64_t num_ids = inverse_unique_partition_indices->shape_view().elem_cnt();
    const int64_t parallel_num = ctx->parallel_ctx().parallel_num();
    const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
    const int64_t padded_embedding_size =
        data_shuffle::GetPaddedEmbeddingSize(data_type, embedding_size);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    ncclComm_t comm = kernel_state->comm();
    using ComputeType = typename DefaultComputeType<T>::type;
    bool enable_quantized_comm_env_var =
        ParseBooleanFromEnv("ONEFLOW_ONE_EMBEDDING_ENABLE_QUANTIZED_COMM", false);
    bool enable_quantized_comm =
        enable_quantized_comm_env_var && (padded_embedding_size < kMaxColSize);
    if (enable_quantized_comm_env_var && !enable_quantized_comm) {
      LOG(WARNING) << "Only envrionment variable ONEFLOW_ONE_EMBEDDING_ENABLE_QUANTIZED_COMM=1 and "
                      "embedding_size less equal than 1024 can use quantized communication. ";
    }
    const bool skip_first_scatter = ctx->Attr<bool>("skip_first_scatter");
    cudaStream_t cuda_stream = ctx->stream()->As<ep::CudaStream>()->cuda_stream();
    const std::vector<uint32_t>& num_unique_matrix_vec =
        embedding_state->GetIdNumUniqueMatrix(current_iter_);
    CHECK_EQ(sizeof(IDX), sizeof(uint32_t)) << "assume sizeof(IDX) equals to sizeof(uint32_t)";
    std::memcpy(host_num_unique_matrix, num_unique_matrix_vec.data(),
                parallel_num * parallel_num * sizeof(IDX));
    uint32_t num_unique = embedding_state->GetIdNumUnique(current_iter_);

    int64_t cur_rank_num_ids = 0;
    for (int64_t i = 0; i < parallel_num; ++i) {
      cur_rank_num_ids += host_num_unique_matrix[i * parallel_num + parallel_id];
    }
    int64_t unique_partitioned_num_ids = 0;
    for (int64_t i = 0; i < parallel_num; ++i) {
      unique_partitioned_num_ids += host_num_unique_matrix[parallel_id * parallel_num + i];
    }
    if (!enable_quantized_comm) {
      // 1. sum to unique grad, from (num_ids, embedding_size) to (unique_partitioned_num_ids,
      // padded_embedding_size)
      void* unique_partition_embedding_grad;  // T
      allocator->Allocate(&unique_partition_embedding_grad,
                          unique_partitioned_num_ids * padded_embedding_size * sizeof(T));

      const T* unique_embedding_grad_ptr;
      if (skip_first_scatter) {
        unique_embedding_grad_ptr = embedding_grad->dptr<T>();
      } else {
        data_shuffle::UniquePartitionEmbeddingGrad(
            ctx->stream(), unique_partitioned_num_ids, num_ids, embedding_size,
            padded_embedding_size, host_num_unique_matrix, embedding_grad->dptr<T>(),
            reinterpret_cast<const IDX*>(inverse_unique_partition_indices->dptr()),
            reinterpret_cast<T*>(unique_partition_embedding_grad));
        unique_embedding_grad_ptr = reinterpret_cast<T*>(unique_partition_embedding_grad);
      }
      // 2. send recv grad, from (unique_partitioned_num_ids, padded_embedding_size) to
      // (cur_rank_num_ids, padded_embedding_size)
      void* received_embedding_grad;  // T
      allocator->Allocate(&received_embedding_grad,
                          cur_rank_num_ids * padded_embedding_size * sizeof(T));

      data_shuffle::ShuffleEmbeddingsGrad(cuda_stream, comm, parallel_id, parallel_num, num_ids,
                                          padded_embedding_size, data_type, host_num_unique_matrix,
                                          unique_embedding_grad_ptr,
                                          reinterpret_cast<T*>(received_embedding_grad));

      // 3. sum to unique grad, from (cur_rank_num_ids, padded_embedding_size) to (num_unique,
      // padded_embedding_size) then slice to out from (num_unique, padded_embedding_size) to
      // (num_unique, embedding_size) should memset cur_rank_unique_embedding_grad all tensor for
      // amp count_not_finite
      // use unique_partition_embedding_grad as UniqueCurRankEmbeddingGrad buffer.
      T* buffer_ptr = reinterpret_cast<T*>(unique_partition_embedding_grad);
      data_shuffle::UniqueCurRankEmbeddingGrad<T, IDX>(
          ctx->stream(), data_type, cur_rank_num_ids, num_unique, embedding_size,
          padded_embedding_size, only_zero_valid_grad,
          cur_rank_unique_embedding_grad->shape_view().elem_cnt(),
          reinterpret_cast<T*>(received_embedding_grad),
          reinterpret_cast<const IDX*>(cur_rank_inverse_indices->dptr()),
          cur_rank_unique_embedding_grad->mut_dptr<T>(), buffer_ptr);
      allocator->Free(unique_partition_embedding_grad);
      allocator->Free(received_embedding_grad);
    } else {
      CHECK(!skip_first_scatter) << "when enable_quantized_comm, should not use fuse kernel.";
      // 1. sum to unique grad, from (num_ids, embedding_size) to (unique_partitioned_num_ids,
      // padded_embedding_size)
      void* unique_partition_embedding_grad;  // T
      allocator->Allocate(&unique_partition_embedding_grad,
                          unique_partitioned_num_ids * padded_embedding_size * sizeof(T));

      data_shuffle::UniquePartitionEmbeddingGrad(
          ctx->stream(), unique_partitioned_num_ids, num_ids, embedding_size, padded_embedding_size,
          host_num_unique_matrix, embedding_grad->dptr<T>(),
          reinterpret_cast<const IDX*>(inverse_unique_partition_indices->dptr()),
          reinterpret_cast<T*>(unique_partition_embedding_grad));

      // 2. Quantize unique_partition_embedding_grad, get
      // quantize_cur_rank_embedding_grad(unique_partitioned_num_ids, padded_embedding_size) int8_t
      // and cur_rank_quantize_factor(unique_partitioned_num_ids) T
      void* quantize_cur_rank_embedding_grad;  // int8_t
      allocator->Allocate(&quantize_cur_rank_embedding_grad,
                          unique_partitioned_num_ids * padded_embedding_size * sizeof(int8_t));
      void* cur_rank_quantize_factor;  // T
      allocator->Allocate(&cur_rank_quantize_factor, unique_partitioned_num_ids * sizeof(T));

      DispatchQuantizeWarpImplPackSize<T, ComputeType>()(
          cuda_stream, reinterpret_cast<T*>(unique_partition_embedding_grad),
          reinterpret_cast<int8_t*>(quantize_cur_rank_embedding_grad),
          reinterpret_cast<T*>(cur_rank_quantize_factor), unique_partitioned_num_ids,
          padded_embedding_size);

      // 3. send recv grad, from (unique_partitioned_num_ids, padded_embedding_size) int8_t to
      // (cur_rank_num_ids, padded_embedding_size) int8_t send recv quantize_factor, from
      // (unique_partitioned_num_ids) T to (cur_rank_num_ids) T
      void* received_embedding_grad;  // int8_t
      allocator->Allocate(&received_embedding_grad,
                          cur_rank_num_ids * padded_embedding_size * sizeof(int8_t));
      void* received_cur_rank_quantize_factor;  // T
      allocator->Allocate(&received_cur_rank_quantize_factor, cur_rank_num_ids * sizeof(T));

      data_shuffle::ShuffleEmbeddingsGrad(
          cuda_stream, comm, parallel_id, parallel_num, num_ids, padded_embedding_size, data_type,
          host_num_unique_matrix, reinterpret_cast<int8_t*>(quantize_cur_rank_embedding_grad),
          reinterpret_cast<int8_t*>(received_embedding_grad),
          reinterpret_cast<T*>(cur_rank_quantize_factor),
          reinterpret_cast<T*>(received_cur_rank_quantize_factor));
      allocator->Free(quantize_cur_rank_embedding_grad);
      allocator->Free(cur_rank_quantize_factor);

      /*
      Host num unique matrix:
              |  Partition0  |  Partition1  |
      | Rank0 |      2       |       4      |
      | Rank1 |      3       |       3      |
      After ShuffleEmbeddingGrads, each rank will exchange partition.
      For example:
      Rank0 will have (matrix[rank0][part0] + matrix[rank1][part0]) grad tensor.
      Rank1 will have (matrix[rank0][part1] + matrix[rank1][part1]) grad tensor.
      */
      // 4. dequantize grad, from (cur_rank_num_ids, padded_embedding_size) int8_t to
      // (cur_rank_num_ids, padded_embedding_size) T
      void* dequantize_cur_rank_embedding_grad;  // T
      allocator->Allocate(&dequantize_cur_rank_embedding_grad,
                          cur_rank_num_ids * padded_embedding_size * sizeof(T));

      OF_CUDA_CHECK((LaunchDequantizeKernel<T, ComputeType, IDX>(
          cuda_stream, reinterpret_cast<int8_t*>(received_embedding_grad),
          reinterpret_cast<T*>(received_cur_rank_quantize_factor),
          reinterpret_cast<T*>(dequantize_cur_rank_embedding_grad), padded_embedding_size,
          cur_rank_num_ids * padded_embedding_size)));
      allocator->Free(received_embedding_grad);
      allocator->Free(received_cur_rank_quantize_factor);

      // use unique_partition_embedding_grad as UniqueCurRankEmbeddingGrad buffer.
      T* buffer_ptr = reinterpret_cast<T*>(unique_partition_embedding_grad);
      // 5. sum to unique grad, from (cur_rank_num_ids, padded_embedding_size) to (num_unique,
      // padded_embedding_size) then slice to out from (num_unique, padded_embedding_size) to
      // (num_unique, embedding_size) should memset cur_rank_unique_embedding_grad all tensor for
      // amp count_not_finite
      data_shuffle::UniqueCurRankEmbeddingGrad<T, IDX>(
          ctx->stream(), data_type, cur_rank_num_ids, num_unique, embedding_size,
          padded_embedding_size, only_zero_valid_grad,
          cur_rank_unique_embedding_grad->shape_view().elem_cnt(),
          reinterpret_cast<T*>(dequantize_cur_rank_embedding_grad),
          reinterpret_cast<const IDX*>(cur_rank_inverse_indices->dptr()),
          cur_rank_unique_embedding_grad->mut_dptr<T>(), buffer_ptr);
      allocator->Free(unique_partition_embedding_grad);
      allocator->Free(dequantize_cur_rank_embedding_grad);
    }
    current_iter_++;
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  mutable int64_t current_iter_;
};

#define REGISTER_CUDA_EMBEDDING_GRADIENT_SHUFFLE_KERNEL(t_dtype_pair, idx_dtype_pair)             \
  REGISTER_USER_KERNEL("embedding_gradient_shuffle")                                              \
      .SetCreateFn<EmbeddingGradientShuffleKernel<OF_PP_PAIR_FIRST(t_dtype_pair),                 \
                                                  OF_PP_PAIR_FIRST(idx_dtype_pair)>>()            \
      .SetIsMatchedHob(                                                                           \
          (user_op::HobDeviceType() == DeviceType::kCUDA)                                         \
          && (user_op::HobDataType("embedding_grad", 0) == OF_PP_PAIR_SECOND(t_dtype_pair))       \
          && ((user_op::HobAttr<bool>("skip_first_scatter") == false)                             \
              || (!embedding::UseEmbeddingGradientShuffleP2PKernel(                               \
                  OF_PP_PAIR_SECOND(t_dtype_pair), OF_PP_PAIR_SECOND(idx_dtype_pair))))           \
          && (user_op::HobDataType("num_unique_matrix", 0) == OF_PP_PAIR_SECOND(idx_dtype_pair))) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                         \
        const user_op::TensorDesc& cur_rank_unique_embedding_grad =                               \
            ctx->InputTensorDesc("cur_rank_unique_embedding_grad", 0);                            \
        size_t cur_rank_embedding_grad_num = cur_rank_unique_embedding_grad.shape().At(0);        \
        size_t embedding_size = cur_rank_unique_embedding_grad.shape().At(1);                     \
        size_t padded_embedding_size = data_shuffle::GetPaddedEmbeddingSize(                      \
            cur_rank_unique_embedding_grad.data_type(), embedding_size);                          \
        size_t cur_rank_embedding_grad_elem_cnt =                                                 \
            cur_rank_embedding_grad_num * padded_embedding_size;                                  \
        bool enable_quantized_comm =                                                              \
            ParseBooleanFromEnv("ONEFLOW_ONE_EMBEDDING_ENABLE_QUANTIZED_COMM", false)             \
            && (padded_embedding_size < kMaxColSize);                                             \
        size_t tmp_size = 0;                                                                      \
        if (embedding::UseDynamicMemoryAllocation()) { return tmp_size; }                         \
        if (!enable_quantized_comm) {                                                             \
          size_t cur_rank_embedding_grad_size = GetCudaAlignedSize(                               \
              cur_rank_embedding_grad_elem_cnt * sizeof(OF_PP_PAIR_FIRST(t_dtype_pair)));         \
          tmp_size = 2 * cur_rank_embedding_grad_size;                                            \
        } else {                                                                                  \
          size_t unique_partition_embedding_grad_size = GetCudaAlignedSize(                       \
              cur_rank_embedding_grad_elem_cnt * sizeof(OF_PP_PAIR_FIRST(t_dtype_pair)));         \
          size_t received_embedding_grad_size =                                                   \
              GetCudaAlignedSize(cur_rank_embedding_grad_elem_cnt * sizeof(int8_t));              \
          size_t quantize_cur_rank_embedding_grad_size = received_embedding_grad_size;            \
          size_t cur_rank_quantize_factor_size = GetCudaAlignedSize(                              \
              cur_rank_embedding_grad_num * sizeof(OF_PP_PAIR_FIRST(t_dtype_pair)));              \
          size_t received_cur_rank_quantize_factor_size = cur_rank_quantize_factor_size;          \
          size_t dequantize_cur_rank_embedding_grad_size = unique_partition_embedding_grad_size;  \
          tmp_size = unique_partition_embedding_grad_size + received_embedding_grad_size          \
                     + quantize_cur_rank_embedding_grad_size + cur_rank_quantize_factor_size      \
                     + received_cur_rank_quantize_factor_size                                     \
                     + dequantize_cur_rank_embedding_grad_size;                                   \
        }                                                                                         \
        return tmp_size;                                                                          \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_CUDA_EMBEDDING_GRADIENT_SHUFFLE_KERNEL,
                                 FLOATING_DATA_TYPE_SEQ HALF_DATA_TYPE_SEQ, IDX_DATA_TYPE_SEQ)

template<typename IDX>
class EmbeddingUniqueKeyValuePairKernelState final : public user_op::OpKernelState {
 public:
  explicit EmbeddingUniqueKeyValuePairKernelState(user_op::KernelInitContext* ctx)
      : device_index_(-1) {
    OF_CUDA_CHECK(cudaGetDevice(&device_index_));
    OF_CUDA_CHECK(cudaMallocHost(&host_num_keys_, sizeof(IDX)));
    const std::string& embedding_name = ctx->Attr<std::string>("embedding_name");
    const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
    embedding_state_ = Singleton<embedding::EmbeddingManager>::Get()->GetEmbeddingState(
        embedding_name, parallel_id);
  }
  ~EmbeddingUniqueKeyValuePairKernelState() {
    CudaCurrentDeviceGuard guard(device_index_);
    OF_CUDA_CHECK(cudaFreeHost(host_num_keys_));
  }

  embedding::EmbeddingState* EmbeddingState() { return embedding_state_; }

  IDX* HostNumKeys() { return host_num_keys_; }

 private:
  int device_index_;
  embedding::EmbeddingState* embedding_state_;
  IDX* host_num_keys_;
};

template<typename K, typename V, typename IDX>
class UniqueKeyValuePairKernel final : public user_op::OpKernel {
 public:
  UniqueKeyValuePairKernel() : current_iter_(0){};
  ~UniqueKeyValuePairKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<EmbeddingUniqueKeyValuePairKernelState<IDX>>(ctx);
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    auto* kernel_state = dynamic_cast<EmbeddingUniqueKeyValuePairKernelState<IDX>*>(state);
    CHECK(kernel_state != nullptr);
    const user_op::Tensor* keys = ctx->Tensor4ArgNameAndIndex("keys", 0);
    user_op::Tensor* num_unique = ctx->Tensor4ArgNameAndIndex("num_unique", 0);
    user_op::Tensor* unique_keys = ctx->Tensor4ArgNameAndIndex("unique_keys", 0);
    user_op::Tensor* unique_values = ctx->Tensor4ArgNameAndIndex("unique_values", 0);
    user_op::Tensor* inverse_indices = ctx->Tensor4ArgNameAndIndex("inverse_indices", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const int32_t num_tables = ctx->Attr<int32_t>("num_tables");
    const int64_t padding_idx = ctx->Attr<int64_t>("padding_idx");
    const bool has_padding_idx = ctx->Attr<bool>("has_padding_idx");
    const bool has_values = ctx->has_input("values", 0);
    const bool need_values_buffer = (!has_values && num_tables > 1);
    size_t values_buffer_bytes =
        need_values_buffer ? GetCudaAlignedSize(keys->shape_view().elem_cnt() * sizeof(V)) : 0;
    const int64_t num_keys = keys->shape_view().elem_cnt();
    const int64_t hash_capacity = num_keys;
    const size_t workspace_bytes =
        GetCudaAlignedSize(hash_capacity * sizeof(data_shuffle::TableEntry<K>));
    CHECK_LE(values_buffer_bytes + workspace_bytes, tmp_buffer->shape_view().elem_cnt());
    cudaStream_t cuda_stream = ctx->stream()->As<ep::CudaStream>()->cuda_stream();
    const V* values_ptr;
    if (has_values) {
      const user_op::Tensor* values = ctx->Tensor4ArgNameAndIndex("values", 0);
      values_ptr = reinterpret_cast<const V*>(values->dptr());
    } else if (need_values_buffer) {
      V* values_buffer_ptr = reinterpret_cast<V*>(tmp_buffer->mut_dptr());
      data_shuffle::GenerateTableIds<<<BlocksNum4ThreadsNum(num_keys), kCudaThreadsNumPerBlock, 0,
                                       cuda_stream>>>(num_keys, num_tables, values_buffer_ptr);
      values_ptr = values_buffer_ptr;
    } else {
      values_ptr = nullptr;
    }
    const bool need_process_table_ids = (has_values || num_tables > 1);
    data_shuffle::TableEntry<K>* workspace_ptr = reinterpret_cast<data_shuffle::TableEntry<K>*>(
        tmp_buffer->mut_dptr<char>() + values_buffer_bytes);
    data_shuffle::UniqueAndPartition<K, V, IDX, embedding::GlobalUniqueHash>(
        cuda_stream, num_keys, hash_capacity, 1, reinterpret_cast<const K*>(keys->dptr()),
        values_ptr, reinterpret_cast<IDX*>(num_unique->mut_dptr()),
        reinterpret_cast<K*>(unique_keys->mut_dptr()),
        reinterpret_cast<V*>(unique_values->mut_dptr()),
        reinterpret_cast<IDX*>(inverse_indices->mut_dptr()), workspace_ptr, workspace_bytes,
        need_process_table_ids, has_padding_idx, padding_idx);

    IDX* host_num_keys = kernel_state->HostNumKeys();
    OF_CUDA_CHECK(cudaMemcpyAsync(host_num_keys, num_unique->mut_dptr(), sizeof(IDX),
                                  cudaMemcpyDefault, cuda_stream));
    CHECK_JUST(ctx->stream()->Sync());
    uint32_t num_unique_ids = *host_num_keys;
    embedding::EmbeddingState* embedding_state = kernel_state->EmbeddingState();
    std::vector<uint32_t> num_unique_matrix_vec({num_unique_ids});
    embedding_state->SetIdNumUniqueMatrix(num_unique_matrix_vec, current_iter_);
    embedding_state->SetIdFinalNumUnique(num_unique_ids, current_iter_);
    current_iter_++;
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  mutable int64_t current_iter_;
};

#define REGISTER_CUDA_UNIQUE_KEY_VALUE_PAIR_KERNEL(k_dtype_pair, value_dtype_pair, idx_dtype_pair) \
  REGISTER_USER_KERNEL("unique_key_value_pair")                                                    \
      .SetCreateFn<UniqueKeyValuePairKernel<OF_PP_PAIR_FIRST(k_dtype_pair),                        \
                                            OF_PP_PAIR_FIRST(value_dtype_pair),                    \
                                            OF_PP_PAIR_FIRST(idx_dtype_pair)>>()                   \
      .SetIsMatchedHob(                                                                            \
          (user_op::HobDeviceType() == DeviceType::kCUDA)                                          \
          && (user_op::HobDataType("keys", 0) == OF_PP_PAIR_SECOND(k_dtype_pair))                  \
          && (user_op::HobDataType("inverse_indices", 0) == OF_PP_PAIR_SECOND(idx_dtype_pair))     \
          && (user_op::HobDataType("unique_values", 0) == OF_PP_PAIR_SECOND(value_dtype_pair)))    \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                          \
        const user_op::TensorDesc& keys = ctx->InputTensorDesc("keys", 0);                         \
        const int64_t num_keys = keys.shape().elem_cnt();                                          \
        const int64_t hash_capacity = num_keys;                                                    \
        const size_t workspace_bytes = GetCudaAlignedSize(                                         \
            hash_capacity * sizeof(data_shuffle::TableEntry<OF_PP_PAIR_FIRST(k_dtype_pair)>));     \
        const int32_t num_tables = ctx->Attr<int32_t>("num_tables");                               \
        const bool has_values = ctx->has_input("values", 0);                                       \
        const bool need_values_buffer = (!has_values && num_tables > 1);                           \
        size_t values_buffer_bytes =                                                               \
            need_values_buffer                                                                     \
                ? GetCudaAlignedSize(num_keys * sizeof(OF_PP_PAIR_FIRST(value_dtype_pair)))        \
                : 0;                                                                               \
        return workspace_bytes + values_buffer_bytes;                                              \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_CUDA_UNIQUE_KEY_VALUE_PAIR_KERNEL, ID_DATA_TYPE_SEQ,
                                 TABLE_ID_DATA_TYPE_SEQ, IDX_DATA_TYPE_SEQ)

template<typename T, typename IDX>
class OneEmbeddingGatherKernel final : public user_op::OpKernel {
 public:
  OneEmbeddingGatherKernel() : current_iter_(0) {}
  ~OneEmbeddingGatherKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<EmbeddingUniqueKeyValuePairKernelState<IDX>>(ctx);
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    auto* kernel_state = dynamic_cast<EmbeddingUniqueKeyValuePairKernelState<IDX>*>(state);
    CHECK(kernel_state != nullptr);
    embedding::EmbeddingState* embedding_state = kernel_state->EmbeddingState();
    embedding_state->OnEmbeddingGatherStart(ctx, current_iter_);
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* indices = ctx->Tensor4ArgNameAndIndex("indices", 0);
    const int64_t num_indices = indices->shape_view().elem_cnt();
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    uint32_t num_unique = embedding_state->GetIdNumUnique(current_iter_);
    const int64_t embedding_size = ctx->Attr<int64_t>("embedding_size");
    const T* in_ptr = reinterpret_cast<const T*>(embedding_state->EmbeddingGatherIn(current_iter_));
    GatherKernelUtilImpl<DeviceType::kCUDA, T, IDX>::Forward(
        ctx->stream(), reinterpret_cast<const IDX*>(indices->dptr()), num_indices, in_ptr,
        Shape({1, num_unique, embedding_size}), out->mut_dptr<T>(), 0);
    embedding_state->OnEmbeddingGatherEnd(ctx, current_iter_);
    current_iter_++;
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  mutable int64_t current_iter_;
};

#define REGISTER_ONE_EMBEDDING_GATHER_KERNEL(in_type, indices_type)                              \
  REGISTER_USER_KERNEL("one_embedding_gather")                                                   \
      .SetCreateFn<                                                                              \
          OneEmbeddingGatherKernel<OF_PP_PAIR_FIRST(in_type), OF_PP_PAIR_FIRST(indices_type)>>() \
      .SetIsMatchedHob(                                                                          \
          (user_op::HobDeviceType() == DeviceType::kCUDA)                                        \
          && (user_op::HobDataType("in", 0) == OF_PP_PAIR_SECOND(in_type))                       \
          && (user_op::HobDataType("indices", 0) == OF_PP_PAIR_SECOND(indices_type)));

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_ONE_EMBEDDING_GATHER_KERNEL,
                                 FLOATING_DATA_TYPE_SEQ HALF_DATA_TYPE_SEQ, IDX_DATA_TYPE_SEQ)

REGISTER_USER_KERNEL_UNIFIED_NCCL_COMM_INIT("id_shuffle");
REGISTER_USER_KERNEL_UNIFIED_NCCL_COMM_INIT("embedding_shuffle");
REGISTER_USER_KERNEL_UNIFIED_NCCL_COMM_INIT("embedding_gradient_shuffle");

}  // namespace oneflow
