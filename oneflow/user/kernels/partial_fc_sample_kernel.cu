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
#ifdef WITH_CUDA

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/kernel/gather_kernel_util.h"
#include "oneflow/core/common/not_equal_to_previous_adjacent_iterator.h"
#include <cub/cub.cuh>
#include <curand.h>
#include <curand_kernel.h>

namespace oneflow {
namespace user_op {

namespace {

template<typename K>
int64_t GetCubSortPairsTempStorageSize(int64_t n) {
  size_t cub_sort_temp_store_size = 0;
  OF_CUDA_CHECK((cub::DeviceRadixSort::SortPairs<K, K>(nullptr, cub_sort_temp_store_size, nullptr,
                                                       nullptr, nullptr, nullptr, n)));
  size_t temp_store_size = GetCudaAlignedSize(static_cast<int64_t>(cub_sort_temp_store_size));
  CHECK_GE(temp_store_size, 0);
  CHECK_LT(temp_store_size, GetMaxVal<int64_t>());
  return temp_store_size;
}

template<typename K>
void CubSortPairs(cudaStream_t stream, int64_t n, size_t temp_storage_bytes, const K* keys,
                  const K* vals, void* tmp_storage, K* sorted_keys, K* sorted_vals) {
  OF_CUDA_CHECK(
      (cub::DeviceRadixSort::SortPairs<K, K>(tmp_storage, temp_storage_bytes, keys, sorted_keys,
                                             vals, sorted_vals, n, 0, sizeof(K) * 8, stream)));
}

template<typename K>
int64_t GetCubScanTempStorageSize(int64_t n) {
  size_t cub_scan_temp_store_size = 0;
  NotEqualToPreviousAdjacentIterator<K, K> unique_counting_iter(nullptr, 0);
  OF_CUDA_CHECK((cub::DeviceScan::InclusiveSum<NotEqualToPreviousAdjacentIterator<K, K>, K*>(
      nullptr, cub_scan_temp_store_size, unique_counting_iter, nullptr, n)));
  size_t temp_store_size = GetCudaAlignedSize(static_cast<int64_t>(cub_scan_temp_store_size));
  CHECK_GE(temp_store_size, 0);
  CHECK_LT(temp_store_size, GetMaxVal<int64_t>());
  return temp_store_size;
}

template<typename K>
class TmpBufferManager final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TmpBufferManager);
  TmpBufferManager(void* ptr, const int64_t device_num_class, const int64_t batch_size)
      : ptr_(ptr) {
    const int64_t buffer_elem_cnt = std::max(device_num_class, batch_size);
    const size_t cub_sort_keys_bytes = GetCudaAlignedSize(buffer_elem_cnt * sizeof(K));
    const size_t cub_sort_values_bytes = GetCudaAlignedSize(buffer_elem_cnt * sizeof(K));
    const size_t cub_sort_keys_out_bytes = GetCudaAlignedSize(buffer_elem_cnt * sizeof(K));
    const size_t cub_sort_values_out_bytes = GetCudaAlignedSize(buffer_elem_cnt * sizeof(K));
    cub_tmp_storage_bytes_ = std::max(GetCubSortPairsTempStorageSize<K>(buffer_elem_cnt),
                                      GetCubScanTempStorageSize<K>(batch_size));
    cub_sort_keys_offset_ = 0;
    cub_sort_values_offset_ = cub_sort_keys_offset_ + cub_sort_keys_bytes;
    cub_sort_keys_out_offset_ = cub_sort_values_offset_ + cub_sort_keys_bytes;
    cub_sort_values_out_offset_ = cub_sort_keys_out_offset_ + cub_sort_keys_out_bytes;
    cub_tmp_storage_offset_ = cub_sort_values_out_offset_ + cub_sort_values_out_bytes;
    total_buffer_size_ = cub_sort_keys_bytes + cub_sort_values_bytes + cub_sort_keys_out_bytes
                         + cub_sort_values_out_bytes + cub_tmp_storage_bytes_;
  }
  ~TmpBufferManager() = default;

  size_t GetTotalBufferSize() const { return total_buffer_size_; }
  size_t GetCubTmpStorageSize() const { return cub_tmp_storage_bytes_; }
  K* CubSortKeysPtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<K*>(reinterpret_cast<char*>(ptr_) + cub_sort_keys_offset_);
  }
  K* CubSortValuesPtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<K*>(reinterpret_cast<char*>(ptr_) + cub_sort_values_offset_);
  }
  K* CubSortKeysOutPtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<K*>(reinterpret_cast<char*>(ptr_) + cub_sort_keys_out_offset_);
  }
  K* CubSortValuesOutPtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<K*>(reinterpret_cast<char*>(ptr_) + cub_sort_values_out_offset_);
  }
  void* CubTmpStoragePtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<void*>(reinterpret_cast<char*>(ptr_) + cub_tmp_storage_offset_);
  }

 private:
  size_t cub_sort_keys_offset_;
  size_t cub_sort_values_offset_;
  size_t cub_sort_keys_out_offset_;
  size_t cub_sort_values_out_offset_;
  size_t cub_tmp_storage_offset_;
  size_t cub_tmp_storage_bytes_;
  size_t total_buffer_size_;
  void* ptr_;
};

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
                                          int64_t num_sample_per_rank, int64_t seed)
      : lower_(lower), upper_(upper), num_sample_per_rank_(num_sample_per_rank) {
    CHECK_NOTNULL(ctx);
    const int64_t num_classes = upper_ - lower_;
    OF_CUDA_CHECK(cudaMalloc(&curand_states_, BlocksNum4ThreadsNum(num_classes)
                                                  * kCudaThreadsNumPerBlock * sizeof(curandState)));
    SetupKernel<<<BlocksNum4ThreadsNum(num_classes), kCudaThreadsNumPerBlock, 0,
                  ctx->cuda_stream()>>>(seed, curand_states_);
  }
  ~DistributedPartialFcSampleOpKernelState() { OF_CUDA_CHECK(cudaFree(curand_states_)); };

  int64_t lower() const { return lower_; }
  int64_t upper() const { return upper_; }
  int64_t num_sample_per_rank() const { return num_sample_per_rank_; }

  template<typename K>
  void GenRandomIndexs(DeviceCtx* ctx, const int64_t n, const int64_t max_val, K* buffer) {
    GenerateGpu<K><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        curand_states_, n, max_val, buffer);
  }

 private:
  const int64_t lower_;
  const int64_t upper_;
  const int64_t num_sample_per_rank_;
  curandState* curand_states_;
};

template<typename K>
__global__ void IotaKernel(int64_t n, K* out) {
  CUDA_1D_KERNEL_LOOP(i, n) { out[i] = static_cast<K>(i); }
}

template<typename K>
__global__ void IndexSetPos(const int64_t n, const int64_t offset, const int64_t num_classes,
                            const K* labels, K* out) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    K label = labels[i] - offset;
    if (label >= 0 && label < num_classes) { out[label] = -1; }
  }
}

template<typename K>
__global__ void GetSampleLabel(const int64_t n, const int64_t offset, const K* label,
                               K* sample_label) {
  CUDA_1D_KERNEL_LOOP(i, n) { sample_label[i] = label[i] + offset; }
}

template<typename K>
__global__ void GetLabelMap(const int64_t n, const int64_t parallel_num,
                            const int64_t num_sample_per_rank, const K* bound_index,
                            const K* bound_value, K* label_map) {
  CUDA_1D_KERNEL_LOOP(i, n) {
#pragma unroll
    for (int64_t j = 0; j < parallel_num; j++) {
      if (i >= bound_index[j] && i < bound_index[j + 1]) {
        label_map[i] = label_map[i] - bound_value[j] + j * num_sample_per_rank;
      }
    }
  }
}

template<typename K>
__global__ void GetPartionBound(const int64_t n, const int64_t parallel_num,
                                const int64_t num_classes_per_rank, const K* key_ptr,
                                const K* value_ptr, K* bound_index, K* bound_value) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    if (i != 0 && i != n - 1) {
      const K cur_in = key_ptr[i];
      const K pre_in = key_ptr[i - 1];
#pragma unroll
      for (int32_t j = 1; j < parallel_num; ++j) {
        const int32_t lower_bound = j * num_classes_per_rank;
        if (cur_in >= lower_bound && pre_in < lower_bound) {
          bound_index[j] = static_cast<K>(i);
          bound_value[j] = value_ptr[i];
        }
      }
    }
    if (i == 0) {
      const K in = key_ptr[i];
#pragma unroll
      for (int32_t j = 0; j <= parallel_num; ++j) {
        const int32_t lower_bound = j * num_classes_per_rank;
        if (in >= lower_bound) {
          bound_index[j] = 0;
          bound_value[j] = value_ptr[i];
        }
      }
    }
    if (i == n - 1) {
      const K in = key_ptr[i];
#pragma unroll
      for (int32_t j = parallel_num; j >= 0; --j) {
        const int32_t lower_bound = j * num_classes_per_rank;
        if (in < lower_bound) {
          bound_index[j] = n;
          bound_value[j] = value_ptr[i];
        }
      }
    }
  }
}

template<typename K>
__global__ void GetMappedLabel(const int64_t n, const K* label_map_key, const K* label_map_value,
                               K* maped_label) {
  CUDA_1D_KERNEL_LOOP(i, n) { maped_label[label_map_key[i]] = label_map_value[i]; }
}

template<typename K>
void SampleIndex(DeviceCtx* ctx, const int64_t num_classes, const int64_t batch_size,
                 const int64_t lower_bound, const TmpBufferManager<K>& buffer_manager,
                 const K* label_ptr) {
  IotaKernel<<<BlocksNum4ThreadsNum(num_classes), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
      num_classes, buffer_manager.CubSortValuesPtr());
  IndexSetPos<<<BlocksNum4ThreadsNum(batch_size), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
      batch_size, lower_bound, num_classes, label_ptr, buffer_manager.CubSortKeysPtr());
  CubSortPairs<K>(ctx->cuda_stream(), num_classes, buffer_manager.GetCubTmpStorageSize(),
                  buffer_manager.CubSortKeysPtr(), buffer_manager.CubSortValuesPtr(),
                  buffer_manager.CubTmpStoragePtr(), buffer_manager.CubSortKeysOutPtr(),
                  buffer_manager.CubSortValuesOutPtr());
}

template<typename K>
void MapLabel(DeviceCtx* ctx, const int64_t num_classes, const int64_t batch_size,
              const int64_t lower_bound, const int64_t parallel_num, const int64_t num_sample,
              const TmpBufferManager<K>& buffer_manager, const K* label_ptr, K* maped_label_ptr) {
  IotaKernel<<<BlocksNum4ThreadsNum(batch_size), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
      batch_size, buffer_manager.CubSortValuesPtr());
  CubSortPairs<K>(ctx->cuda_stream(), batch_size, buffer_manager.GetCubTmpStorageSize(), label_ptr,
                  buffer_manager.CubSortValuesPtr(), buffer_manager.CubTmpStoragePtr(),
                  buffer_manager.CubSortKeysOutPtr(), buffer_manager.CubSortValuesOutPtr());

  size_t temp_storage_bytes = buffer_manager.GetCubTmpStorageSize();
  NotEqualToPreviousAdjacentIterator<K, K> unique_counting_iter(buffer_manager.CubSortKeysOutPtr(),
                                                                0);
  OF_CUDA_CHECK((cub::DeviceScan::InclusiveSum<NotEqualToPreviousAdjacentIterator<K, K>, K*>(
      buffer_manager.CubTmpStoragePtr(), temp_storage_bytes, unique_counting_iter,
      buffer_manager.CubSortValuesPtr(), batch_size, ctx->cuda_stream())));

  K* bound_index = buffer_manager.CubSortKeysPtr();
  K* bound_value = buffer_manager.CubSortKeysPtr() + parallel_num + 1;
  GetPartionBound<<<BlocksNum4ThreadsNum(batch_size), kCudaThreadsNumPerBlock, 0,
                    ctx->cuda_stream()>>>(
      batch_size, parallel_num, num_classes, buffer_manager.CubSortKeysOutPtr(),
      buffer_manager.CubSortValuesPtr(), bound_index, bound_value);

  GetLabelMap<K>
      <<<BlocksNum4ThreadsNum(batch_size), kCudaThreadsNumPerBlock, parallel_num * sizeof(K),
         ctx->cuda_stream()>>>(batch_size, parallel_num, num_sample, bound_index, bound_value,
                               buffer_manager.CubSortValuesPtr());

  GetMappedLabel<<<BlocksNum4ThreadsNum(batch_size), kCudaThreadsNumPerBlock, 0,
                   ctx->cuda_stream()>>>(batch_size, buffer_manager.CubSortValuesOutPtr(),
                                         buffer_manager.CubSortValuesPtr(), maped_label_ptr);
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
    const int64_t seed = ctx->Attr<int64_t>("seed");
    const int64_t parallel_num = ctx->parallel_ctx().parallel_num();
    const int64_t num_sample_per_rank = RoundUp(num_sample, parallel_num) / parallel_num;
    if (in_sbp.has_split_parallel() && in_sbp.split_parallel().axis() == 0 && parallel_num > 1) {
      CHECK(ctx->SbpParallel4ArgNameAndIndex("label", 0).has_broadcast_parallel());
      BalancedSplitter bs(class_num, parallel_num);
      return std::make_shared<DistributedPartialFcSampleOpKernelState>(
          ctx->device_ctx(), bs.At(ctx->parallel_ctx().parallel_id()).begin(),
          bs.At(ctx->parallel_ctx().parallel_id()).end(), num_sample_per_rank, seed);
    } else {
      return std::make_shared<DistributedPartialFcSampleOpKernelState>(
          ctx->device_ctx(), 0, class_num, num_sample_per_rank, seed);
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
    const int64_t parallel_num = ctx->parallel_ctx().parallel_num();
    TmpBufferManager<K> buffer_manager(tmp_buffer->mut_dptr(), num_classes, batch_size);

    auto* kernel_state = dynamic_cast<DistributedPartialFcSampleOpKernelState*>(state);
    CHECK_NOTNULL(kernel_state);
    CHECK_EQ(num_classes, kernel_state->upper() - kernel_state->lower());
    const int64_t lower_bound = kernel_state->lower();
    const int64_t num_sample = kernel_state->num_sample_per_rank();
    kernel_state->GenRandomIndexs<K>(ctx->device_ctx(), num_classes, num_classes,
                                     buffer_manager.CubSortKeysPtr());
    SampleIndex<K>(ctx->device_ctx(), num_classes, batch_size, lower_bound, buffer_manager,
                   label->dptr<K>());

    GetSampleLabel<<<BlocksNum4ThreadsNum(num_sample), kCudaThreadsNumPerBlock, 0,
                     ctx->device_ctx()->cuda_stream()>>>(num_sample, lower_bound,
                                                         buffer_manager.CubSortValuesOutPtr(),
                                                         sampled_label->mut_dptr<K>());

    GatherKernelUtilImpl<DeviceType::kGPU, T, K>::Forward(
        ctx->device_ctx(), buffer_manager.CubSortValuesOutPtr(), num_sample, weight->dptr<T>(),
        Shape({1, weight->shape().At(0), weight->shape().Count(1)}), sampled_weight->mut_dptr<T>(),
        0);

    MapLabel<K>(ctx->device_ctx(), num_classes, batch_size, lower_bound, parallel_num, num_sample,
                buffer_manager, label->dptr<K>(), maped_label->mut_dptr<K>());
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
        const int64_t batch_size = ctx->TensorDesc4ArgNameAndIndex("label", 0)->shape().At(0);   \
        TmpBufferManager<OF_PP_PAIR_FIRST(ltype_pair)> buffer_manager(nullptr, num_classes,      \
                                                                      batch_size);               \
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
#endif
