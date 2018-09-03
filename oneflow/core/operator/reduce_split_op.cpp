#include "oneflow/core/operator/reduce_split_op.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/register/runtime_blob_desc.h"

namespace oneflow {

void ReduceSplitOp::InitFromOpConf() {
  CHECK(op_conf().has_reduce_split_conf());
  for (int32_t i = 0; i < op_conf().reduce_split_conf().out_num(); ++i) {
    EnrollOutputBn("out_" + std::to_string(i), false);
  }
  EnrollInputBn("in", false);
}

const PbMessage& ReduceSplitOp::GetCustomizedConf() const { return op_conf().reduce_split_conf(); }

void ReduceSplitOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*,
    KernelConf* kernel_conf) const {
  ReduceSplitKernelConf* reduce_split_conf = kernel_conf->mutable_reduce_split_conf();
  int64_t offset = 0;
  for (int32_t i = 0; i < op_conf().reduce_split_conf().out_num(); ++i) {
    reduce_split_conf->mutable_data_offset()->Add(offset);
    offset += RtBlobDesc(*(GetBlobDesc4BnInOp(output_bns().Get(i)))).ByteSizeOfBlobBody();
  }
  CHECK_EQ(offset, RtBlobDesc(*GetBlobDesc4BnInOp(SoleIbn())).ByteSizeOfBlobBody());
}

REGISTER_OP(OperatorConf::kReduceSplitConf, ReduceSplitOp);

}  // namespace oneflow
