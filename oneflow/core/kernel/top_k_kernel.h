#ifndef ONEFLOW_CORE_KERNEL_TOP_K_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_TOP_K_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class TopKKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TopKKernel);
  TopKKernel() = default;
  ~TopKKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<typename T>
void CpuTopK(DeviceCtx* ctx, const T* in, int32_t* indices, int32_t instance_num,
             int32_t instance_size, int32_t k, bool sorted, int32_t* out);

template<typename T>
void GpuHeapSelectionTopK(DeviceCtx* ctx, const T* in, int32_t instance_num, int32_t instance_size,
                          int32_t k, int32_t* out);

template<typename T>
void GpuRadixSortTopK(DeviceCtx* ctx, const T* in, int32_t* indices, int32_t instance_num,
                      int32_t instance_size, int32_t k, void* temp_storage,
                      size_t temp_storage_bytes, T* sorted_in, int32_t* sorted_indices,
                      int32_t* out);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_TOP_K_KERNEL_H_
