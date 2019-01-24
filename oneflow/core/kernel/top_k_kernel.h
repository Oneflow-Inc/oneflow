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
  static void Forward(const T* in_ptr, const bool sotred, const int32_t instance_num,
                      const int32_t instance_size, const int32_t k, int32_t* fw_buf,
                      int32_t* out_ptr);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_TOP_K_KERNEL_H_
