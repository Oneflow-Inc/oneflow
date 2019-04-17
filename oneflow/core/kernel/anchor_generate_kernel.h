#ifndef ONEFLOW_CORE_OPERATOR_ANCHOR_GENERATE_KERNEL_OP_H_
#define ONEFLOW_CORE_OPERATOR_ANCHOR_GENERATE_KERNEL_OP_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/bbox_util.h"

namespace oneflow {

template<typename T>
class AnchorGenerateKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AnchorGenerateKernel);
  AnchorGenerateKernel() = default;
  ~AnchorGenerateKernel() = default;

  using BBox = BBoxT<const T>;
  using MutBBox = BBoxT<T>;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void ForwardDim0ValidNum(const KernelCtx&,
                           std::function<Blob*(const std::string&)>) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_ANCHOR_GENERATE_KERNEL_OP_H_
