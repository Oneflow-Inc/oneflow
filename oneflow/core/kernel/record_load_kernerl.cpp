#include "oneflow/core/kernel/record_load_kernel.h"

namespace oneflow {

void RecordLoadKernel::Forward(const KernelCtx& ctx,
                               std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  UNIMPLEMENTED();
}

REGISTER_KERNEL(OperatorConf::kRecordLoadConf, RecordLoadKernel);

}  // namespace oneflow
