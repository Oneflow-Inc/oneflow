#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/operator/repeat_op.h"

namespace oneflow {

void oneflow::RepeatOp::InitFromOpConf() {
  CHECK(op_conf().has_repeat_conf());
  const RepeatOpConf& conf = op_conf().repeat_conf();
  if (conf.has_repeat_num()) {
    CHECK_GE(conf.repeat_num(), 1);
  } else if (conf.has_repeat_num_per_record()) {
    CHECK_GE(conf.repeat_num_per_record(), 1);
  } else {
    UNIMPLEMENTED();
  }
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

void RepeatOp::InferOutBlobTimeShape(
    std::function<const Shape*(const std::string&)> GetTimeShape4BnInOp,
    const ParallelContext* parallel_ctx, Shape* time_shape) const {
  std::vector<int64_t> dim_vec(GetTimeShape4BnInOp("in")->dim_vec());
  int32_t repeat_num = GetRepeatNum(parallel_ctx->parallel_num());
  dim_vec.push_back(repeat_num);
  *time_shape = Shape(dim_vec);
}

int32_t RepeatOp::GetRepeatNum(int64_t parallel_num) const {
  CHECK(op_conf().has_repeat_conf());
  const RepeatOpConf& conf = op_conf().repeat_conf();
  if (conf.has_repeat_num()) {
    return conf.repeat_num();
  } else if (conf.has_repeat_num_per_record()) {
    CHECK_EQ(Global<JobDesc>::Get()->PieceSize() % parallel_num, 0);
    int64_t repeat_num =
        Global<JobDesc>::Get()->PieceSize() / parallel_num * conf.repeat_num_per_record();
    CHECK_LE(repeat_num, static_cast<int64_t>(GetMaxVal<int32_t>()));
    return static_cast<int32_t>(repeat_num);
  } else {
    UNIMPLEMENTED();
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
