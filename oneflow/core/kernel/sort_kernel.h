#ifndef ONEFLOW_CORE_OPERATOR_ARG_SORT_KERNEL_OP_H_
#define ONEFLOW_CORE_OPERATOR_ARG_SORT_KERNEL_OP_H_

#include "oneflow/core/kernel/kernel.h"

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

template<typename T>
void CpuSort(DeviceCtx* ctx, const T* in_ptr, int32_t instance_num, int32_t instance_size,
             std::string dir, T* out_ptr);

template<typename T>
void GpuSort(DeviceCtx* ctx, const T* in_ptr, int32_t instance_num, int32_t instance_size,
             std::string dir, void* temp_storage_ptr, size_t temp_storage_bytes, T* out_ptr);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_ARG_SORT_KERNEL_OP_H_
