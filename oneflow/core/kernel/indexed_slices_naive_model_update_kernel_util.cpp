#include "oneflow/core/kernel/indexed_slices_naive_model_update_kernel_util.h"

namespace oneflow {

template<typename T, typename K>
struct IndexedSlicesNaiveMdUpdateKernelUtil<DeviceType::kCPU, T, K> final {
  static void Update(DeviceCtx* ctx, const K* indices, const T* values, const float* learning_rate,
                     int64_t num_indices, int64_t num_features, int64_t feature_size,
                     int64_t feature_id_offset, T* model);
};

template<typename T, typename K>
void IndexedSlicesNaiveMdUpdateKernelUtil<DeviceType::kCPU, T, K>::Update(
    DeviceCtx* ctx, const K* indices, const T* values, const float* learning_rate,
    int64_t num_indices, int64_t num_features, int64_t feature_size, int64_t feature_id_offset,
    T* model) {
  FOR_RANGE(int64_t, i, 0, num_indices) {
    const K feature_id = indices[i];
    CHECK_GE(feature_id, 0);
    const K local_feature_id = feature_id - feature_id_offset;
    if (local_feature_id >= 0 && local_feature_id < num_features) {
      const T* from = values + i * feature_size;
      T* to = model + local_feature_id * feature_size;
      for (int64_t j = 0; j < feature_size; ++j) { to[j] -= from[j] * (*learning_rate); }
    }
  }
}
#define INITIATE_INDEXED_SLICES_NAIVE_MODEL_UPDATE_KERNEL_UTIL_GPU(in_type_pair, index_type_pair) \
  template struct IndexedSlicesNaiveMdUpdateKernelUtil<                                           \
      DeviceType::kCPU, OF_PP_PAIR_FIRST(in_type_pair), OF_PP_PAIR_FIRST(index_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INITIATE_INDEXED_SLICES_NAIVE_MODEL_UPDATE_KERNEL_UTIL_GPU,
                                 FLOATING_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ);
#undef INITIATE_INDEXED_SLICES_NAIVE_MODEL_UPDATE_KERNEL_UTIL_GPU

}  // namespace oneflow
