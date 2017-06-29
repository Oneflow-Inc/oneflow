#ifndef ONEFLOW_CORE_KERNEL_INNERPRODUCT_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_INNERPRODUCT_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_manager.h"
#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

template<DeviceType device_type, typename floating_point_type>
class InnerProductKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(InnerProductKernel);
  InnerProductKernel() = default;
  ~InnerProductKernel() = default;

  void InitFromOpProto(const OperatorProto& op_proto) override;

  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override;
  void Backward(const KernelCtx&,
                std::function<Blob*(const std::string&)>) const override;
 
 protected:
  void InitModelAndModelTmpBlobsWithoutSnapshot(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;

 private:
  bool has_bias_term_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_INNERPRODUCT_KERNEL_H_
