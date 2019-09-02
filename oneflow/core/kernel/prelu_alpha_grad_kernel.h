#ifndef ONEFLOW_CORE_KERNEL_PRELU_ALPHA_GRAD_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_PRELU_ALPHA_GRAD_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class PReluAlphaGradKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PReluAlphaGradKernel);
  PReluAlphaGradKernel() = default;
  ~PReluAlphaGradKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
struct PReluAlphaGradKernelUtil {
  static void Compute(const KernelCtx& ctx, const PReluAlphaGradOpConf& conf,
                      const PbRf<int32_t>& permutation, const Blob* x_blob, const Blob* dy_blob,
                      Blob* bw_buf_blob, Blob* alpha_grad_buf_blob, Blob* alpha_grad_blob);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_PRELU_ALPHA_GRAD_KERNEL_H_
