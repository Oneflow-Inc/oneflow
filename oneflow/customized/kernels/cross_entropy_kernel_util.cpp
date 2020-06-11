#include "oneflow/customized/kernels/sparse_cross_entropy_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"

namespace oneflow {
namespace user_op {

template<typename T, typename K>
struct CrossEntropyKernelUtil<DeviceType::kCPU, T, K> {
  static void ComputeEntropy(DeviceCtx* ctx, const int64_t num_instances, const int64_t num_classes,
                             const T* x, const K* labels, T* y) {
    FOR_RANGE(int64_t, i, 0, num_instances) {
      T tmp = 0;
      FOR_RANGE(int64_t, j, 0, num_classes) {
        K label = labels[i * num_classes + j];
        T prob = x[i * num_classes + j];
        // CPU SafeLog crashes on my side
        // tmp -= label * SafeLog(prob);
        tmp -= label * logf((prob > 1e-20) ? prob : 1e-20);
      }
      y[i] = tmp;
    }
  }

  static void ComputeDiffWithSoftmax(DeviceCtx* ctx, const int64_t elem_cnt,
                                     const int64_t num_classes, const T* prob, const K* labels,
                                     const T* dy, T* dx) {
    FOR_RANGE(int64_t, i, 0, elem_cnt) {
      const int32_t row_id = i / num_classes;
      dx[i] = dy[row_id] * (prob[i] - labels[i]);
    }
  }
};

#define INSTANTIATE_CROSS_ENTROPY_KERNEL_UTIL_CPU(data_type_pair, index_type_pair)           \
  template struct CrossEntropyKernelUtil<DeviceType::kCPU, OF_PP_PAIR_FIRST(data_type_pair), \
                                         OF_PP_PAIR_FIRST(index_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_CROSS_ENTROPY_KERNEL_UTIL_CPU, FLOATING_DATA_TYPE_SEQ,
                                 FLOATING_DATA_TYPE_SEQ);
#undef INSTANTIATE_CROSS_ENTROPY_KERNEL_UTIL_CPU

}  // namespace user_op
}  // namespace oneflow
