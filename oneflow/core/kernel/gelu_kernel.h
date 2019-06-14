#ifndef ONEFLOW_CORE_KERNEL_GELU_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_GELU_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class GeluKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GeluKernel);
  GeluKernel() = default;
  ~GeluKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  const PbMessage& GetCustomizedOpConf() const override;
};

template<DeviceType device_type, typename T>
struct GeluKernelUtil {
  static void GeluForward(DeviceCtx* ctx, const int64_t n, const T* x, T* y);

  static void GeluBackward(DeviceCtx* ctx, const int64_t n, const T* x, const T* dy, T* dx);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_GELU_KERNEL_H_
