#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) {
  CHECK(op.op_conf().has_concat_conf());
  const ConcatOpConf& concat_conf = op.op_conf().concat_conf();
  OperatorConf split_op;
  split_op.set_name(op.op_conf().name() + "_grad");
  SplitLikeOpConf* split_like_op_conf = split_op.mutable_split_like_conf();
  split_like_op_conf->set_in(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
  split_like_op_conf->set_axis(concat_conf.axis());
  FOR_RANGE(int32_t, i, 0, concat_conf.in_size()) {
    const std::string& ibn_of_concat_op = op.input_bns().Get(i);
    const std::string& obn = "out_" + i;
    split_like_op_conf->add_like(GenLogicalBlobName(op.BnInOp2Lbi(ibn_of_concat_op)));
    split_like_op_conf->add_out(obn);
    if (DiffLbi4BnInOp(ibn_of_concat_op) != nullptr) {
      DiffLbi4BnInOp(ibn_of_concat_op)->set_op_name(split_op.name());
      DiffLbi4BnInOp(ibn_of_concat_op)->set_blob_name(obn);
    }
  }
  op_confs->push_back(split_op);
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kConcatConf, &GenerateBackwardOpConf);

}  // namespace oneflow
