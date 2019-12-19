#ifndef ONEFLOW_CORE_KERNEL_BIAS_ADD_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_BIAS_ADD_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class BiasAddKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BiasAddKernel);
  BiasAddKernel() = default;
  ~BiasAddKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  const PbMessage& GetCustomizedOpConf() const override;
};

template<DeviceType device_type, typename T>
struct BiasAddUtil {
  static void BiasAdd(DeviceCtx* ctx, int64_t outer_size, int64_t bias_size, int64_t inner_size,
                      const T* x, const T* bias, T* y);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BIAS_ADD_KERNEL_H_
