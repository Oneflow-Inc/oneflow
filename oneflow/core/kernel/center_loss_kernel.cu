#include "oneflow/core/kernel/center_loss_kernel.h"

namespace oneflow {

template<typename PredType, typename LabelType>
struct CenterLossKernelUtil<DeviceType::kGPU, PredType, LabelType> {
  static void Gather(const PredType* centers_ptr, const LabelType* label_ptr,
                     PredType* piece_centers_ptr) {
    // TODO
  }
  static void SparseUpdate(int32_t n, const LabelType* label_ptr, PredType* center_diff_ptr,
                           PredType* centers_ptr) {
    // TODO
  }
};

}  // namespace oneflow