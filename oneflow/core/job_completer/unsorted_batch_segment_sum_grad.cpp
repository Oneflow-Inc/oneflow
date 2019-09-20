#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_unsorted_batch_segment_sum_conf());
  if (DiffLbi4BnInOp("data") != nullptr) {
    OperatorConf batch_gather;
    batch_gather.set_name("System-AutoGrad-" + op.op_name());
    BatchGatherOpConf* conf = batch_gather.mutable_batch_gather_conf();
    conf->set_in(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    conf->set_indices(GenLogicalBlobName(op.BnInOp2Lbi("segment_ids")));
    conf->set_out("out");
    op_confs->push_back(batch_gather);
    DiffLbi4BnInOp("data")->set_op_name(batch_gather.name());
    DiffLbi4BnInOp("data")->set_blob_name(conf->out());
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kUnsortedBatchSegmentSumConf, &GenerateBackwardOpConf);

}  // namespace oneflow
