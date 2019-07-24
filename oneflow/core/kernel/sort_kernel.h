#ifndef ONEFLOW_CORE_OPERATOR_SORT_KERNEL_OP_H_
#define ONEFLOW_CORE_OPERATOR_SORT_KERNEL_OP_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/bbox_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class SortKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SortKernel);
  SortKernel() = default;
  ~SortKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
struct SortUtil {
  static void Forward(DeviceCtx* ctx, const T* key_ptr, const int32_t* value_ptr,
                      void* temp_storage_ptr, size_t temp_storage_bytes, int32_t num_row,
                      int32_t num_col, T* sorted_key_ptr, int32_t* sorted_value_ptr);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_SORT_KERNEL_OP_H_
