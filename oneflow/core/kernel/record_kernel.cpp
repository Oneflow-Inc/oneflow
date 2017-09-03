#include "oneflow/core/kernel/record_kernel.h"

namespace oneflow {

void RecordKernel::Forward(
    const KernelCtx& kernel_ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // TODO
}

}  // namespace oneflow
