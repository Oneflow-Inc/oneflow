#ifndef ONEFLOW_CORE_OPERATOR_ARG_SORT_KERNEL_OP_H_
#define ONEFLOW_CORE_OPERATOR_ARG_SORT_KERNEL_OP_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class ArgSortKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ArgSortKernel);
  ArgSortKernel() = default;
  ~ArgSortKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<typename T>
void CpuArgSort(DeviceCtx* ctx, const T* in_ptr, int32_t instance_num, int32_t instance_size,
                std::string dir, int32_t* out_ptr);

template<typename T>
void GpuArgSort(DeviceCtx* ctx, const T* in_ptr, int32_t* indices_ptr, int32_t instance_num,
                int32_t instance_size, std::string dir, void* temp_storage_ptr,
                size_t temp_storage_bytes, T* sorted_in_ptr, int32_t* out_ptr);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_ARG_SORT_KERNEL_OP_H_
