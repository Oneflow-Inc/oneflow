#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

Maybe<void> GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) {
  CHECK(op.op_conf().has_batch_gather_conf());
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf unsorted_batch_segment_sum;
    unsorted_batch_segment_sum.set_name("System-AutoGrad-" + op.op_name());
    auto* conf = unsorted_batch_segment_sum.mutable_unsorted_batch_segment_sum_conf();
    conf->set_data(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    conf->set_segment_ids(GenLogicalBlobName(op.BnInOp2Lbi("indices")));
    conf->set_out("out");
    const Shape& in_shape = LogicalBlobDesc4BnInOp("in").shape();
    const Shape& indices_shape = LogicalBlobDesc4BnInOp("indices").shape();
    CHECK_GE(in_shape.NumAxes(), indices_shape.NumAxes());
    CHECK_GE(indices_shape.NumAxes(), 1);
    conf->set_num_segments(in_shape.At(indices_shape.NumAxes() - 1));
    op_confs->push_back(unsorted_batch_segment_sum);
    DiffLbi4BnInOp("in")->set_op_name(unsorted_batch_segment_sum.name());
    DiffLbi4BnInOp("in")->set_blob_name(conf->out());
  }
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kBatchGatherConf, &GenerateBackwardOpConf);

}  // namespace oneflow
