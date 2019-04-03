#include "oneflow/core/kernel/model_save_v2_kernel.h"

namespace oneflow {

void ModelSaveV2Kernel::VirtualKernelInit(const ParallelContext* parallel_ctx) {}

void ModelSaveV2Kernel::Forward(const KernelCtx& ctx,
                                std::function<Blob*(const std::string&)> BnInOp2Blob) const {}

REGISTER_KERNEL(OperatorConf::kModelSaveV2Conf, ModelSaveV2Kernel);

}  // namespace oneflow
