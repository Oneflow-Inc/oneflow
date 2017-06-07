#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

void Kernel::InitModelAndModelTmpBlobs(
    const KernelContext& ctx,
    std::function<Blob*(const std::string&)> Blob4BnInOp) const {
  TODO();
}

} // namespace oneflow
