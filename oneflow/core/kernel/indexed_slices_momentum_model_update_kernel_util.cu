#include "oneflow/core/kernel/indexed_slices_momentum_model_update_kernel_util.h"
#include "oneflow/core/kernel/unique_kernel_util.h"

namespace oneflow {

namespace {

template<typename T, typename K, typename IDX>
__global__ void UpdateGpu(T beta, int64_t feature_size, int64_t lower_bound, int64_t upper_bound,
                          const IDX* num_unique_instance, const float* learning_rate,
                          const K* indices, const T* values, T* model, T* momentum) {
  const int64_t n = *num_unique_instance * feature_size;
  const float lr = *learning_rate;
  CUDA_1D_KERNEL_LOOP(i, n) {
    const K instance_id = indices[i / feature_size];
    if (instance_id >= lower_bound && instance_id < upper_bound) {
      const T diff = values[i];
      const K model_idx = (instance_id - lower_bound) * feature_size + i % feature_size;
      const T next_momentum = beta * momentum[model_idx] - lr * diff;
      momentum[model_idx] = next_momentum;
      model[model_idx] = model[model_idx] + next_momentum;
    }
  }
}

}  // namespace

template<typename T, typename K, typename IDX>
struct IndexedSlicesMomentumMdUpdateKernelUtil<DeviceType::kGPU, T, K, IDX> {
  static void Update(DeviceCtx* ctx, T beta, int64_t num_instance, int64_t feature_size,
                     int64_t lower_bound, int64_t upper_bound, const IDX* num_unique_instance,
                     const int64_t* train_step, const float* learning_rate, const K* indices,
                     const T* values, T* model, T* momentum);
};

template<typename T, typename K, typename IDX>
void IndexedSlicesMomentumMdUpdateKernelUtil<DeviceType::kGPU, T, K, IDX>::Update(
    DeviceCtx* ctx, T beta, int64_t num_instance, int64_t feature_size, int64_t lower_bound,
    int64_t upper_bound, const IDX* num_unique_instance, const int64_t* train_step,
    const float* learning_rate, const K* indices, const T* values, T* model, T* momentum) {
  UpdateGpu<T, K><<<BlocksNum4ThreadsNum(num_instance * feature_size), kCudaThreadsNumPerBlock, 0,
                    ctx->cuda_stream()>>>(beta, feature_size, lower_bound, upper_bound,
                                          num_unique_instance, learning_rate, indices, values,
                                          model, momentum);
}

#define INSTANTIATE_INDEXED_SLICES_MOMENTUM_MODEL_UPDATE_KERNEL_UTIL_GPU(                 \
    val_type_pair, key_type_pair, idx_type_pair)                                          \
  template struct IndexedSlicesMomentumMdUpdateKernelUtil<                                \
      DeviceType::kGPU, OF_PP_PAIR_FIRST(val_type_pair), OF_PP_PAIR_FIRST(key_type_pair), \
      OF_PP_PAIR_FIRST(idx_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_INDEXED_SLICES_MOMENTUM_MODEL_UPDATE_KERNEL_UTIL_GPU,
                                 FLOATING_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ);
#undef INSTANTIATE_INDEXED_SLICES_MOMENTUM_MODEL_UPDATE_KERNEL_UTIL_GPU

}  // namespace oneflow
