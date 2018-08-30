#ifndef ONEFLOW_CORE_KERNEL_PROPOSAL_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_PROPOSAL_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/faster_rcnn_util.h"

namespace oneflow {

template<typename T>
class ProposalKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ProposalKernel);
  ProposalKernel() = default;
  ~ProposalKernel() = default;

 private:
  void ForwardDataId(const KernelCtx&, std::function<Blob*(const std::string&)>) const override;
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void InitConstBufBlobs(DeviceCtx*,
                         std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void CopyRoI(const int64_t im_index, const ScoredBoxesIndex<T>& boxes, Blob* rois_blob) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_PROPOSAL_KERNEL_H_
