#ifndef ONEFLOW_CORE_KERNEL_POOLING_GRAD_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_POOLING_GRAD_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/device/cudnn_util.h"
#include "oneflow/core/common/eigen_util.h"
#include "oneflow/core/kernel/pooling_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class PoolingGradKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PoolingGradKernel);
  PoolingGradKernel() = default;
  ~PoolingGradKernel() = default;

 private:
  const PbMessage& GetCustomizedOpConf() const override;
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
struct PoolingGradKernelUtil final {
  static void Compute(DeviceCtx* ctx, const PoolingConf& pooling_conf, const Blob* dy_blob,
                      const Blob* y_blob, const Blob* x_blob, Blob* dx_blob);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_POOLING_GRAD_KERNEL_H_
