#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/operator/repeat_op.h"

namespace oneflow {

void RepeatOp::InitFromOpConf() {
  CHECK(op_conf().has_repeat_conf());
  const RepeatOpConf& conf = op_conf().repeat_conf();
  CHECK_GE(conf.repeat_num(), 1);
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

void RepeatOp::InferOutputBlobTimeShape(
    std::function<const Shape*(const std::string&)> GetTimeShape4BnInOp,
    const ParallelContext* parallel_ctx, Shape* time_shape) const {
  std::vector<int64_t> dim_vec(GetTimeShape4BnInOp("in")->dim_vec());
  int32_t repeat_num = GetRepeatNum();
  dim_vec.push_back(repeat_num);
  *time_shape = Shape(dim_vec);
}

int32_t RepeatOp::GetRepeatNum() const {
  CHECK(op_conf().has_repeat_conf());
  const RepeatOpConf& conf = op_conf().repeat_conf();
  return conf.repeat_num();
}

const PbMessage& RepeatOp::GetCustomizedConf() const { return op_conf().repeat_conf(); }

void RepeatOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                              const ParallelContext* parallel_ctx) const {
  BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *in_blob_desc;
}

LogicalNode* RepeatOp::NewProperLogicalNode() const { return new RepeatForwardLogicalNode(); }

REGISTER_OP(OperatorConf::kRepeatConf, RepeatOp);

}  // namespace oneflow
