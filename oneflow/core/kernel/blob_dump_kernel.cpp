#include "oneflow/core/kernel/blob_dump_kernel.h"

namespace oneflow {

void BlobDumpKernel::VirtualKernelInit(const ParallelContext* parallel_ctx) {
  const BlobDumpOpConf& conf = op_conf().blob_dump_conf();
  const std::string& root_path = conf.dir();
  if (root_path.empty()) {
    base_dir_ = JoinPath("/tmp/" + op_conf().name(), std::to_string(parallel_ctx->parallel_id()));
  } else {
    base_dir_ = JoinPath(root_path, std::to_string(parallel_ctx->parallel_id()));
  }
  iter_ = 0;
}

void BlobDumpKernel::Forward(const KernelCtx& ctx,
                             std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  PersistentOutStream out(SnapshotFS(), JoinPath(base_dir_, std::to_string(iter_)));
  const Blob* in = BnInOp2Blob("in");
  out.Write(in->dptr<char>(), in->ByteSizeOfDataContentField());
  out.Flush();
  iter_ += 1;
}

REGISTER_KERNEL(OperatorConf::kBlobDumpConf, BlobDumpKernel);

}  // namespace oneflow
