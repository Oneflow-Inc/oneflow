#include "oneflow/core/kernel/sparse_cross_entropy_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"

namespace oneflow {

template<typename T, typename K>
struct SparseCrossEntropyKernelUtil<DeviceType::kCPU, T, K> {
  static void ComputeEntropy(DeviceCtx* ctx, int64_t num_instances, K lower_bound,
                             int64_t num_classes, const T* x, const K* labels, T* y) {
    const K upper_bound = lower_bound + num_classes;
    FOR_RANGE(int64_t, i, 0, num_instances) {
      K label = labels[i];
      if (label >= lower_bound && label < upper_bound) {
        y[i] = -SafeLog(x[i * num_classes + label - lower_bound]);
      }
    }
  }

  static void ComputeDiff(DeviceCtx* ctx, int64_t num_instances, K lower_bound, int64_t num_classes,
                          const T* x, const K* labels, T* dx) {
    const K upper_bound = lower_bound + num_classes;
    FOR_RANGE(int64_t, i, 0, num_instances) {
      K label = labels[i];
      if (label >= lower_bound && label < upper_bound) {
        dx[i * num_classes + label - lower_bound] =
            -1 / MaxWithLogThreshold(x[i * num_classes + label - lower_bound]);
      }
    }
  }

  static void ComputeDiff(DeviceCtx* ctx, int64_t num_instances, K lower_bound, int64_t num_classes,
                          const T* x, const K* labels, const T* dy, T* dx) {
    const K upper_bound = lower_bound + num_classes;
    FOR_RANGE(int64_t, i, 0, num_instances) {
      K label = labels[i];
      if (label >= lower_bound && label < upper_bound) {
        dx[i * num_classes + label - lower_bound] =
            -dy[i] / MaxWithLogThreshold(x[i * num_classes + label - lower_bound]);
      }
    }
  }
};

#define INSTANTIATE_SPARSE_CROSS_ENTROPY_KERNEL_UTIL_CPU(data_type_pair, index_type_pair)          \
  template struct SparseCrossEntropyKernelUtil<DeviceType::kCPU, OF_PP_PAIR_FIRST(data_type_pair), \
                                               OF_PP_PAIR_FIRST(index_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_SPARSE_CROSS_ENTROPY_KERNEL_UTIL_CPU,
                                 FLOATING_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ);
#undef INSTANTIATE_SPARSE_CROSS_ENTROPY_KERNEL_UTIL_CPU

}  // namespace oneflow
