#ifndef ONEFLOW_CORE_OPERATOR_BITONIC_SORT_KERNEL_OP_H_
#define ONEFLOW_CORE_OPERATOR_BITONIC_SORT_KERNEL_OP_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/bbox_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class BitonicSortKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BitonicSortKernel);
  BitonicSortKernel() = default;
  ~BitonicSortKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
struct BitonicSortUtil {
  static void Forward(DeviceCtx* ctx, const int32_t instance_num, const int32_t instance_size,
                      T* out);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_BITONIC_SORT_KERNEL_OP_H_
