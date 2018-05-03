#ifndef ONEFLOW_CORE_KERNEL_MODEL_SAVE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_MODEL_SAVE_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

class ModelSaveKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelSaveKernel);
  ModelSaveKernel() = default;
  ~ModelSaveKernel() = default;

 protected:
  void VirtualKernelInit(const ParallelContext*) override;
  void Forward(const KernelCtx&, std::function<Blob*(const std::string&)>) const override;

 private:
  int32_t part_id_;
  int32_t part_num_;
};

using LbiBlobHandler = std::function<void(const LogicalBlobId& lbi, const Blob*)>;
using MdSaveOther = std::tuple<Snapshot*, std::function<void(LbiBlobHandler)>>;

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_MODEL_SAVE_KERNEL_H_
