#ifndef ONEFLOW_CORE_KERNEL_MATMUL_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_MATMUL_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
namespace oneflow {

template<DeviceType device_type, typename T>
class MatmulKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MatmulKernel);
  MatmulKernel() = default;
  ~MatmulKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void Calc2DMatMul(DeviceCtx* ctx, const Blob* a, bool trans_a, const Blob* b, bool trans_b,
                    Blob* c) const;
  void CalcBatchMatMul(DeviceCtx* ctx, const Blob* a, bool trans_a, const Blob* b, bool trans_b,
                       Blob* c, Blob* buf) const;
  const PbMessage& GetCustomizedOpConf() const override;
};

}  // namespace oneflow

#endif  // ONEFLOE_CORE_KERNEL_MATMUL_KERNEL_H_
