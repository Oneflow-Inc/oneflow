#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/operator/repeat_op.h"

namespace oneflow {

void oneflow::RepeatOp::InitFromOpConf() {
  CHECK(op_conf().has_repeat_conf());
  const RepeatOpConf& conf = op_conf().repeat_conf();
  CHECK(conf.has_repeat_num() || conf.has_repeat_num_per_record());
  if (conf.has_repeat_num()) { CHECK_GE(conf.repeat_num(), 1); }
  if (conf.has_repeat_num_per_record()) { CHECK_GE(conf.repeat_num_per_record(), 1); }
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

int32_t RepeatOp::GetRepeatNum(const RepeatOpConf& conf, const ParallelContext& ctx) {
  CHECK(conf.has_repeat_num() || conf.has_repeat_num_per_record());
  if (conf.has_repeat_num()) {
    return conf.repeat_num();
  } else {
    int64_t repeat_num =
        Global<JobDesc>::Get()->DevicePieceSize4ParallelCtx(ctx) * conf.repeat_num_per_record();
    CHECK_LE(repeat_num, static_cast<int64_t>(MaxVal<int32_t>()));
    return static_cast<int32_t>(repeat_num);
  }
}

const PbMessage& RepeatOp::GetCustomizedConf() const { return op_conf().repeat_conf(); }

void RepeatOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                              const ParallelContext* parallel_ctx) const {
  BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *in_blob_desc;
}

void RepeatOp::InferDiffBlobDescsWithoutFwBlob(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  BlobDesc* in_diff_blob_desc = GetBlobDesc4BnInOp(GenDiffBn("in"));
  BlobDesc* out_diff_blob_desc = GetBlobDesc4BnInOp(GenDiffBn("out"));
  *in_diff_blob_desc = *out_diff_blob_desc;
}

LogicalNode* RepeatOp::NewProperLogicalNode() { return new RepeatForwardLogicalNode(); }

REGISTER_OP(OperatorConf::kRepeatConf, RepeatOp);

}  // namespace oneflow
