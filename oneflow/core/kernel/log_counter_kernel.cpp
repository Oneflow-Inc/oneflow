#include "oneflow/core/kernel/log_counter_kernel.h"

namespace oneflow {

void LogCounterKernel::VirtualKernelInit(const ParallelContext* parallel_ctx) {
  counter_.reset(new int64_t(0));
}

void LogCounterKernel::Forward(const KernelCtx& ctx,
                               std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (HasEmptyShapeBlob(this->op_attribute().input_bns(), BnInOp2Blob)) { return; }
  int64_t cnt = (*counter_)++;
  const auto& conf = op_conf().log_counter_conf();
  CHECK_GT(conf.interval(), 0);
  if (cnt % conf.interval() == 0) {
    LOG(INFO) << op_conf().name() << " counter: " << cnt / conf.interval();
  }
}

REGISTER_KERNEL(OperatorConf::kLogCounterConf, LogCounterKernel);

}  // namespace oneflow
