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

template<DeviceType device_type, typename T>
struct TopKKernelUtil {
  static void Forward(const T* prediction_ptr, const int64_t instance_num,
                      const int64_t elem_cnt_per_instance, const int64_t k, T* fw_buf,
                      T* indices_ptr, T* values_ptr);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_TOP_K_KERNEL_H_