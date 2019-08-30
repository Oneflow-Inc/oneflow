#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) {
  CHECK(op.op_conf().has_segment_sum_conf());
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf segment_sum_grad;
    segment_sum_grad.set_name("System-AutoGrad-" + op.op_name());
    SegmentSumGradOpConf* conf = segment_sum_grad.mutable_segment_sum_grad_conf();
    conf->set_out_diff(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    conf->set_segment_ids(GenLogicalBlobName(op.BnInOp2Lbi("segment_ids")));
    conf->set_in_diff("in_diff");
    op_confs->push_back(segment_sum_grad);
    DiffLbi4BnInOp("in")->set_op_name(segment_sum_grad.name());
    DiffLbi4BnInOp("in")->set_blob_name(conf->in_diff());
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kSegmentSumConf, &GenerateBackwardOpConf);

}  // namespace oneflow
