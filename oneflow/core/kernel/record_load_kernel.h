#ifndef ONEFLOW_CORE_KERNEL_RECORD_LOAD_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_RECORD_LOAD_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/persistence/persistent_in_stream.h"

namespace oneflow {

struct RecordLoadStatus {
  bool is_eof;
  int64_t record_num;
};

class RecordLoadKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RecordLoadKernel);
  RecordLoadKernel() = default;
  ~RecordLoadKernel() = default;

 private:
  void VirtualKernelInit(const ParallelContext*) override;
  void Forward(const KernelCtx& ctx,
               std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

  std::unique_ptr<PersistentInStream> in_stream_;
  int64_t piece_size_in_one_loader_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_RECORD_LOAD_KERNEL_H_
