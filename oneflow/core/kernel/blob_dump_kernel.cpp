#include "oneflow/core/kernel/blob_dump_kernel.h"

namespace oneflow {

void BlobDumpKernel::VirtualKernelInit(const ParallelContext* parallel_ctx) {
  iter_ = 0;
  parallel_id_ = parallel_ctx->parallel_id();
}

void BlobDumpKernel::Forward(const KernelCtx& ctx,
                             std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  std::ostringstream oss;
  oss << std::setw(6) << std::setfill('0') << iter_ << "-" << std::setw(6) << std::setfill('0')
      << parallel_id_ << ".bin";
  PersistentOutStream out(SnapshotFS(),
                          JoinPath(this->op_conf().blob_dump_conf().dir(), oss.str()));
  const Blob* in = BnInOp2Blob("in");
  out.Write(in->dptr<char>(), in->ByteSizeOfDataContentField());
  out.Flush();
  iter_ += 1;
}

REGISTER_KERNEL(OperatorConf::kBlobDumpConf, BlobDumpKernel);

}  // namespace oneflow
