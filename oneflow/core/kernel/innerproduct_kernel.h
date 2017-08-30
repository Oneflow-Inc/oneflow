#ifndef ONEFLOW_CORE_KERNEL_INNERPRODUCT_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_INNERPRODUCT_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/kernel/kernel_manager.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class InnerProductKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(InnerProductKernel);
  InnerProductKernel() = default;
  ~InnerProductKernel() = default;

  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override;
  void Backward(const KernelCtx&,
                std::function<Blob*(const std::string&)>) const override;

 private:
  void InitModelBlobsWithRandomSeed(
      const KernelCtx&, std::mt19937,
      std::function<Blob*(const std::string&)>) const override;
  void InitModelBlobsWithSnapshot(
      const KernelCtx& ctx, int32_t part_id, int32_t part_num,
      const Snapshot* snapshot,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void InitModelTmpBlobs(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_INNERPRODUCT_KERNEL_H_
