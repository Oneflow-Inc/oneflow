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
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/kernel/random_generator.h"
#include "oneflow/core/kernel/gather_kernel_util.h"
#include "oneflow/core/kernel/unsorted_segment_sum_kernel_util.h"
#ifdef WITH_CUDA
#include <cub/cub.cuh>
#include <curand.h>
#include <curand_kernel.h>
#endif

namespace oneflow {
namespace user_op {

namespace {

template<typename K>
int64_t GetCubSortPairTempStorageSize(int64_t n) {
  size_t cub_sort_temp_store_size = 0;
  OF_CUDA_CHECK((cub::DeviceRadixSort::SortPairs<K, K>(nullptr, cub_sort_temp_store_size, nullptr,
                                                       nullptr, nullptr, nullptr, n)));
  CHECK_GE(cub_sort_temp_store_size, 0);
  CHECK_LT(cub_sort_temp_store_size, GetMaxVal<int64_t>());
  return GetCudaAlignedSize(static_cast<int64_t>(cub_sort_temp_store_size));
}

template<typename KEY, typename VAL>
void SortPairs(cudaStream_t stream, int64_t n, size_t temp_storage_bytes, const KEY* keys,
               const VAL* vals, void* tmp_storage, KEY* sorted_keys, VAL* sorted_vals) {
  OF_CUDA_CHECK((cub::DeviceRadixSort::SortPairs<KEY, VAL>(tmp_storage, temp_storage_bytes, keys,
                                                           sorted_keys, vals, sorted_vals, n, 0,
                                                           sizeof(KEY) * 8, stream)));
}

template<typename K>
class TmpBufferManager final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TmpBufferManager);
  TmpBufferManager(void* ptr, const int64_t device_num_class) : ptr_(ptr) {
    const size_t label_buffer_bytes = GetCudaAlignedSize(device_num_class * sizeof(K));
    const size_t index_buffer_bytes = GetCudaAlignedSize(device_num_class * sizeof(K));
    const size_t sorted_label_buffer_bytes = GetCudaAlignedSize(device_num_class * sizeof(K));
    const size_t sorted_index_buffer_bytes = GetCudaAlignedSize(device_num_class * sizeof(K));
    const size_t rand_value_bytes = GetCudaAlignedSize(device_num_class * sizeof(unsigned int));
    cub_tmp_storage_bytes_ = GetCubSortPairTempStorageSize<K>(device_num_class);

    label_buffer_offset_ = 0;
    index_buffer_offset_ = label_buffer_offset_ + label_buffer_bytes;
    sorted_label_buffer_offset_ = index_buffer_offset_ + index_buffer_bytes;
    sorted_index_buffer_offset_ = sorted_label_buffer_offset_ + sorted_label_buffer_bytes;
    rand_value_offset_ = sorted_index_buffer_offset_ + sorted_index_buffer_bytes;
    cub_tmp_storage_offset_ = rand_value_offset_ + rand_value_bytes;
    total_buffer_size_ = label_buffer_bytes + index_buffer_bytes + sorted_label_buffer_bytes
                         + sorted_index_buffer_bytes + rand_value_bytes + cub_tmp_storage_bytes_;
  }
  ~TmpBufferManager() = default;

  size_t GetTotalBufferSize() const { return total_buffer_size_; }
  size_t GetCubTmpStorageSize() const { return cub_tmp_storage_bytes_; }
  K* LabelBufferPtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<K*>(reinterpret_cast<char*>(ptr_) + label_buffer_offset_);
  }
  K* IndexBufferPtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<K*>(reinterpret_cast<char*>(ptr_) + index_buffer_offset_);
  }
  K* SortedLabelBufferPtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<K*>(reinterpret_cast<char*>(ptr_) + sorted_label_buffer_offset_);
  }
  K* SortedIndexBufferPtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<K*>(reinterpret_cast<char*>(ptr_) + sorted_index_buffer_offset_);
  }
  unsigned int* RandValuePtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<unsigned int*>(reinterpret_cast<char*>(ptr_) + rand_value_offset_);
  }
  K* LabelMapPtr() const { return LabelBufferPtr(); }
  void* CubTmpStoragePtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<void*>(reinterpret_cast<char*>(ptr_) + cub_tmp_storage_offset_);
  }

 private:
  size_t label_buffer_offset_;
  size_t index_buffer_offset_;
  size_t sorted_label_buffer_offset_;
  size_t sorted_index_buffer_offset_;
  size_t rand_value_offset_;
  size_t cub_tmp_storage_offset_;
  size_t cub_tmp_storage_bytes_;
  size_t total_buffer_size_;
  void* ptr_;
};

int GetThreadNum(const cudaDeviceProp& prop) {
  switch (prop.major) {
    case 3:  // Kepler
      return 2 * 192;
    case 5:  // Maxwell
      return 2 * 128;
    case 6:  // Pascal
      if ((prop.minor == 1) || (prop.minor == 2)) {
        return 2 * 128;
      } else {
        return 2 * 64;
      }
    case 7:  // Volta and Turing
      return 2 * 64;
    default: return 2 * 64;
  }
}

__global__ void SetupKernel(int64_t seed, curandState* state) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  size_t local_seed = (static_cast<size_t>(seed) + 0x9e3779b9U + (static_cast<size_t>(id) << 6U)
                       + (static_cast<size_t>(id) >> 2U));
  curand_init(local_seed, 0, 0, &state[id]);
}

template<typename K>
__global__ void GenerateGpu(curandState* state, const int64_t n, const int64_t max_val, K* buffer) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  curandState localState = state[id];
  CUDA_1D_KERNEL_LOOP(i, n) { buffer[i] = static_cast<K>(curand(state) % max_val); }
  state[id] = localState;
}

class DistributedPartialFcSampleOpKernelState final : public user_op::OpKernelState {
 public:
  DistributedPartialFcSampleOpKernelState(DeviceCtx* ctx, int64_t lower, int64_t upper,
                                          int64_t num_sample_per_rank)
      : lower_(lower), upper_(upper), num_sample_per_rank_(num_sample_per_rank) {
    CHECK_NOTNULL(ctx);

    cudaDeviceProp prop;
    OF_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    block_num_ = prop.multiProcessorCount;
    thread_num_ = GetThreadNum(prop);
    OF_CUDA_CHECK(cudaMalloc(&curand_states_, block_num_ * thread_num_ * sizeof(curandState)));
    SetupKernel<<<block_num_, thread_num_>>>(111L, curand_states_);
  }
  ~DistributedPartialFcSampleOpKernelState() { OF_CUDA_CHECK(cudaFree(curand_states_)); };

  int64_t lower() const { return lower_; }
  int64_t upper() const { return upper_; }
  int64_t num_sample_per_rank() const { return num_sample_per_rank_; }

  template<typename K>
  void GenRandomIndexs(const int64_t n, const int64_t max_val, K* buffer) {
    GenerateGpu<K><<<block_num_, thread_num_>>>(curand_states_, n, max_val, buffer);
  }

 private:
  const int64_t lower_;
  const int64_t upper_;
  const int64_t num_sample_per_rank_;
  curandState* curand_states_;
  int32_t block_num_;
  int32_t thread_num_;
};

template<typename K>
__global__ void InitBuffer(const int64_t n, K* label_buffer) {
  CUDA_1D_KERNEL_LOOP(i, n) { label_buffer[i] = i; }
}

template<typename K>
__global__ void IndexSetPos(const int64_t n, const int64_t offset, const int64_t num_classes,
                            const K* labels, K* index_buffer) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    K label = labels[i] - offset;
    if (label >= 0 && label < num_classes) { index_buffer[label] = -1; }
  }
}

template<typename K>
__global__ void GetLabelMap(const int64_t n, const int64_t map_offset, const K* label,
                            K* label_map) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    K cur_label = label[i];
    label_map[cur_label] = map_offset + i;
  }
}

template<typename K>
__global__ void GetSampleLabel(const int64_t n, const int64_t offset, const K* label,
                               K* sample_label) {
  CUDA_1D_KERNEL_LOOP(i, n) { sample_label[i] = label[i] + offset; }
}

}  // namespace

template<typename T, typename K>
class DistributedPartialFcSampleGpuKernel final : public user_op::OpKernel {
 public:
  DistributedPartialFcSampleGpuKernel() = default;
  ~DistributedPartialFcSampleGpuKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    const SbpParallel& in_sbp = ctx->SbpParallel4ArgNameAndIndex("weight", 0);
    const TensorDesc* in_logical_desc = ctx->LogicalTensorDesc4ArgNameAndIndex("weight", 0);
    const int64_t class_num = in_logical_desc->shape().At(0);
    const int64_t num_sample = ctx->Attr<int64_t>("num_sample");
    const int64_t parallel_num = ctx->parallel_ctx().parallel_num();
    const int64_t num_sample_per_rank = RoundUp(num_sample, parallel_num) / parallel_num;
    if (in_sbp.has_split_parallel() && in_sbp.split_parallel().axis() == 0 && parallel_num > 1) {
      CHECK(ctx->SbpParallel4ArgNameAndIndex("label", 0).has_broadcast_parallel());
      BalancedSplitter bs(class_num, parallel_num);
      return std::make_shared<DistributedPartialFcSampleOpKernelState>(
          ctx->device_ctx(), bs.At(ctx->parallel_ctx().parallel_id()).begin(),
          bs.At(ctx->parallel_ctx().parallel_id()).end(), num_sample_per_rank);
    } else {
      return std::make_shared<DistributedPartialFcSampleOpKernelState>(
          ctx->device_ctx(), 0, class_num, num_sample_per_rank);
    }
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
    const user_op::Tensor* label = ctx->Tensor4ArgNameAndIndex("label", 0);
    user_op::Tensor* maped_label = ctx->Tensor4ArgNameAndIndex("maped_label", 0);
    user_op::Tensor* sampled_label = ctx->Tensor4ArgNameAndIndex("sampled_label", 0);
    user_op::Tensor* sampled_weight = ctx->Tensor4ArgNameAndIndex("sampled_weight", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);

    const int64_t batch_size = label->shape().At(0);
    const int64_t num_classes = weight->shape().At(0);
    TmpBufferManager<K> buffer_manager(tmp_buffer->mut_dptr(), num_classes);

    auto* kernel_state = dynamic_cast<DistributedPartialFcSampleOpKernelState*>(state);
    CHECK_NOTNULL(kernel_state);
    CHECK_EQ(weight->shape().At(0), kernel_state->upper() - kernel_state->lower());
    const int64_t upper_bound = kernel_state->upper();
    const int64_t lower_bound = kernel_state->lower();

    kernel_state->GenRandomIndexs<K>(num_classes, num_classes, buffer_manager.IndexBufferPtr());

    const int64_t num_sample = kernel_state->num_sample_per_rank();
    InitBuffer<<<BlocksNum4ThreadsNum(num_classes), kCudaThreadsNumPerBlock, 0,
                 ctx->device_ctx()->cuda_stream()>>>(num_classes, buffer_manager.LabelBufferPtr());
    IndexSetPos<<<BlocksNum4ThreadsNum(batch_size), kCudaThreadsNumPerBlock, 0,
                  ctx->device_ctx()->cuda_stream()>>>(
        batch_size, lower_bound, num_classes, label->dptr<K>(), buffer_manager.IndexBufferPtr());
    SortPairs<K, K>(ctx->device_ctx()->cuda_stream(), num_classes,
                    buffer_manager.GetCubTmpStorageSize(), buffer_manager.IndexBufferPtr(),
                    buffer_manager.LabelBufferPtr(), buffer_manager.CubTmpStoragePtr(),
                    buffer_manager.SortedIndexBufferPtr(), buffer_manager.SortedLabelBufferPtr());
    // check num_sample > num_pos
    // get sampled_label
    GetSampleLabel<<<BlocksNum4ThreadsNum(num_sample), kCudaThreadsNumPerBlock, 0,
                     ctx->device_ctx()->cuda_stream()>>>(num_sample, lower_bound,
                                                         buffer_manager.SortedLabelBufferPtr(),
                                                         sampled_label->mut_dptr<K>());
    // Memcpy<DeviceType::kGPU>(ctx->device_ctx(), sampled_label->mut_dptr<void>(),
    //                         buffer_manager.SortedLabelBufferPtr(),
    //                         num_sample * GetSizeOfDataType(sampled_label->data_type()));
    // get sampled weight
    GatherKernelUtilImpl<DeviceType::kGPU, T, K>::Forward(
        ctx->device_ctx(), buffer_manager.SortedLabelBufferPtr(), num_sample, weight->dptr<T>(),
        Shape({1, weight->shape().At(0), weight->shape().Count(1)}), sampled_weight->mut_dptr<T>(),
        0);
    // get LabelMap
    const int64_t map_offset = ctx->parallel_ctx().parallel_id() * num_sample;
    GetLabelMap<<<BlocksNum4ThreadsNum(num_sample), kCudaThreadsNumPerBlock, 0,
                  ctx->device_ctx()->cuda_stream()>>>(num_sample, map_offset,
                                                      buffer_manager.SortedLabelBufferPtr(),
                                                      buffer_manager.LabelMapPtr());

    Memset<DeviceType::kGPU>(ctx->device_ctx(), maped_label->mut_dptr(), 0,
                             maped_label->shape().elem_cnt() * sizeof(K));
    GatherKernelUtilImpl<DeviceType::kGPU, K, K>::Forward(
        ctx->device_ctx(), label->dptr<K>(), batch_size, buffer_manager.LabelMapPtr(),
        Shape({1, num_classes, 1}), maped_label->mut_dptr<K>(), lower_bound);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_DISTRIBUTED_PARTIAL_FC_SAMPLE_GPU_KERNEL(dtype_pair, ltype_pair)                \
  REGISTER_USER_KERNEL("distributed_partial_fc_sample")                                          \
      .SetCreateFn<DistributedPartialFcSampleGpuKernel<OF_PP_PAIR_FIRST(dtype_pair),             \
                                                       OF_PP_PAIR_FIRST(ltype_pair)>>()          \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                                        \
                       & (user_op::HobDataType("label", 0) == OF_PP_PAIR_SECOND(ltype_pair))     \
                       & (user_op::HobDataType("weight", 0) == OF_PP_PAIR_SECOND(dtype_pair)))   \
      .SetInferTmpSizeFn([](oneflow::user_op::InferContext* ctx) {                               \
        const int64_t num_classes = ctx->TensorDesc4ArgNameAndIndex("weight", 0)->shape().At(0); \
        TmpBufferManager<OF_PP_PAIR_FIRST(ltype_pair)> buffer_manager(nullptr, num_classes);     \
        return buffer_manager.GetTotalBufferSize();                                              \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_DISTRIBUTED_PARTIAL_FC_SAMPLE_GPU_KERNEL,
                                 FLOATING_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

template<typename T, typename K>
class DistributedPartialFcSampleGradGpuKernel final : public user_op::OpKernel {
 public:
  DistributedPartialFcSampleGradGpuKernel() = default;
  ~DistributedPartialFcSampleGradGpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    const user_op::Tensor* sampled_weight_diff =
        ctx->Tensor4ArgNameAndIndex("sampled_weight_diff", 0);
    const user_op::Tensor* sampled_label = ctx->Tensor4ArgNameAndIndex("sampled_label", 0);
    user_op::Tensor* sampled_weight_diff_out =
        ctx->Tensor4ArgNameAndIndex("sampled_weight_diff_out", 0);
    user_op::Tensor* sampled_label_out = ctx->Tensor4ArgNameAndIndex("sampled_label_out", 0);
    Memcpy<DeviceType::kGPU>(ctx->device_ctx(), sampled_weight_diff_out->mut_dptr<void>(),
                             sampled_weight_diff->dptr<void>(),
                             sampled_weight_diff->shape().elem_cnt()
                                 * GetSizeOfDataType(sampled_weight_diff->data_type()));
    Memcpy<DeviceType::kGPU>(
        ctx->device_ctx(), sampled_label_out->mut_dptr<void>(), sampled_label->dptr<void>(),
        sampled_label->shape().elem_cnt() * GetSizeOfDataType(sampled_label->data_type()));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_DISTRIBUTED_PARTIAL_FC_SAMPLE_GRAD_GPU_KERNEL(dtype_pair, ltype_pair)      \
  REGISTER_USER_KERNEL("distributed_partial_fc_sample_grad")                                \
      .SetCreateFn<DistributedPartialFcSampleGradGpuKernel<OF_PP_PAIR_FIRST(dtype_pair),    \
                                                           OF_PP_PAIR_FIRST(ltype_pair)>>() \
      .SetIsMatchedHob(                                                                     \
          (user_op::HobDeviceTag() == "gpu")                                                \
          & (user_op::HobDataType("sampled_label", 0) == OF_PP_PAIR_SECOND(ltype_pair))     \
          & (user_op::HobDataType("sampled_weight_diff", 0) == OF_PP_PAIR_SECOND(dtype_pair)));
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_DISTRIBUTED_PARTIAL_FC_SAMPLE_GRAD_GPU_KERNEL,
                                 FLOATING_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

}  // namespace user_op
}  // namespace oneflow
