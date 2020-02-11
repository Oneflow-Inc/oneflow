#include "oneflow/core/kernel/indexed_slices_lazy_adam_model_update_kernel_util.h"

namespace oneflow {

template<typename T, typename K, typename IDX>
struct IndexedSlicesLazyAdamMdUpdateKernelUtil<DeviceType::kCPU, T, K, IDX> {
  static void Update(DeviceCtx* ctx, T beta1, T beta2, T epsilon, int64_t num_instance,
                     int64_t feature_size, int64_t lower_bound, int64_t upper_bound,
                     const IDX* num_unique_instance, const int64_t* train_step,
                     const float* learning_rate, const K* indices, const T* values, T* model, T* m,
                     T* v) {
    UNIMPLEMENTED();
  }
  static void ComputeLocalLearningRate(DeviceCtx* ctx, T beta1, T beta2, const int64_t* train_step,
                                       const float* learning_rate, float* local_learning_rate) {
    UNIMPLEMENTED();
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
