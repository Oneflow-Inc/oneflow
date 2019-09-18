#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_unsorted_segment_sum_conf());
  CHECK_GE(op.op_conf().unsorted_segment_sum_conf().axis(), 0);
  if (DiffLbi4BnInOp("data") != nullptr) {
    OperatorConf gather;
    gather.set_name("System-AutoGrad-" + op.op_name());
    GatherOpConf* conf = gather.mutable_gather_conf();
    conf->set_in(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    conf->set_indices(GenLogicalBlobName(op.BnInOp2Lbi("segment_ids")));
    conf->set_axis(op.op_conf().unsorted_segment_sum_conf().axis());
    conf->set_out("out");
    op_confs->push_back(gather);
    DiffLbi4BnInOp("data")->set_op_name(gather.name());
    DiffLbi4BnInOp("data")->set_blob_name(conf->out());
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kUnsortedSegmentSumConf, &GenerateBackwardOpConf);

}  // namespace oneflow
