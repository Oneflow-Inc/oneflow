#include "oneflow/core/kernel/indexed_slices_naive_model_update_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include "oneflow/core/kernel/kernel.h"
#include <assert.h>

namespace oneflow {

namespace {

template<typename T, typename K, typename IDX>
__global__ void UpdateGpu(const IDX data_elem_cnt, const K* indices, const T* values,
                          const float* learning_rate, const IDX num_features,
                          const IDX feature_size, T* model, const IDX feature_id_offset) {
  const T lr = *learning_rate;
  CUDA_1D_KERNEL_LOOP_T(IDX, i, data_elem_cnt) {
    const T val = values[i];
    if (val != static_cast<T>(0)) {
      const IDX feature_id = indices[i / feature_size];
      assert(feature_id >= 0);
      const IDX local_feature_id = feature_id - feature_id_offset;
      if (local_feature_id >= 0 && local_feature_id < num_features) {
        const IDX update_offset = local_feature_id * feature_size + i % feature_size;
        gpu_atomic_add(model + update_offset, -val * lr);
      }
    }
  }
}

}  // namespace

template<typename T, typename K>
struct IndexedSlicesNaiveMdUpdateKernelUtil<DeviceType::kGPU, T, K> final {
  static void Update(DeviceCtx* ctx, const K* indices, const T* values, const float* learning_rate,
                     int64_t num_indices, int64_t num_features, int64_t feature_size,
                     int64_t feature_id_offset, T* model) {
    const int64_t values_elem_cnt = num_indices * feature_size;
    UpdateGpu<T, K, int64_t>
        <<<BlocksNum4ThreadsNum(values_elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            values_elem_cnt, indices, values, learning_rate, num_features, feature_size, model,
            feature_id_offset);
  }
};

#define INITIATE_INDEXED_SLICES_NAIVE_MODEL_UPDATE_KERNEL_UTIL_GPU(in_type_pair, index_type_pair) \
  template struct IndexedSlicesNaiveMdUpdateKernelUtil<                                           \
      DeviceType::kGPU, OF_PP_PAIR_FIRST(in_type_pair), OF_PP_PAIR_FIRST(index_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INITIATE_INDEXED_SLICES_NAIVE_MODEL_UPDATE_KERNEL_UTIL_GPU,
                                 FLOATING_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ);
#undef INITIATE_INDEXED_SLICES_NAIVE_MODEL_UPDATE_KERNEL_UTIL_GPU

}  // namespace oneflow
