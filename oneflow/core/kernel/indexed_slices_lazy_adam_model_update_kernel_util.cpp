#include "oneflow/core/kernel/indexed_slices_lazy_adam_model_update_kernel_util.h"

namespace oneflow {

template<typename T, typename K, typename IDX>
struct IndexedSlicesLazyAdamMdUpdateKernelUtil<DeviceType::kCPU, T, K, IDX> {
  static void Update(DeviceCtx* ctx, T beta1, T beta2, T epsilon, int64_t num_instance,
                     int64_t feature_size, int64_t lower_bound, int64_t upper_bound,
                     const IDX* num_unique_instance, const int64_t* train_step,
                     const float* learning_rate, const K* indices, const T* values, T* model, T* m,
                     T* v) {
    const int64_t n = *num_unique_instance * feature_size;
    for (int64_t i = 0; i < n; ++i) {
      const K instance_id = indices[i / feature_size];
      if (instance_id >= lower_bound && instance_id < upper_bound) {
        const T diff = values[i];
        const K model_idx = (instance_id - lower_bound) * feature_size + i % feature_size;
        const T new_m = beta1 * m[model_idx] + (1 - beta1) * diff;
        const T new_v = beta2 * v[model_idx] + (1 - beta2) * diff * diff;
        m[model_idx] = new_m;
        v[model_idx] = new_v;
        model[model_idx] = model[model_idx] - *learning_rate * new_m / (std::sqrt(new_v) + epsilon);
      }
    }
  }
  static void ComputeLocalLearningRate(DeviceCtx* ctx, T beta1, T beta2, const int64_t* train_step,
                                       const float* learning_rate, float* local_learning_rate) {
    const T beta1_t = std::pow(beta1, *train_step + 1);
    const T beta2_t = std::pow(beta2, *train_step + 1);
    *local_learning_rate = *learning_rate * std::sqrt(1 - (beta2_t)) / (1 - (beta1_t));
  }
};

#define INSTANTIATE_INDEXED_SLICES_LAZY_ADAM_MODEL_UPDATE_KERNEL_UTIL_CPU(                \
    val_type_pair, key_type_pair, idx_type_pair)                                          \
  template struct IndexedSlicesLazyAdamMdUpdateKernelUtil<                                \
      DeviceType::kCPU, OF_PP_PAIR_FIRST(val_type_pair), OF_PP_PAIR_FIRST(key_type_pair), \
      OF_PP_PAIR_FIRST(idx_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_INDEXED_SLICES_LAZY_ADAM_MODEL_UPDATE_KERNEL_UTIL_CPU,
                                 FLOATING_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ);
#undef INSTANTIATE_INDEXED_SLICES_LAZY_ADAM_MODEL_UPDATE_KERNEL_UTIL_CPU

}  // namespace oneflow
