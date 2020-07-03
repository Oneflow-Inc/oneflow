#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

Maybe<void> GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) {
  CHECK(op.op_conf().has_gather_conf());
  if (DiffLbi4BnInOp("in") != nullptr) {
    const BlobDesc& in_logical_blob_desc = LogicalBlobDesc4BnInOp("in");
    const GatherOpConf& gather_conf = op.op_conf().gather_conf();
    const int64_t num_in_axes = in_logical_blob_desc.shape().NumAxes();
    const int64_t axis =
        gather_conf.axis() < 0 ? num_in_axes + gather_conf.axis() : gather_conf.axis();
    CHECK_GE(axis, 0);
    CHECK_LT(axis, num_in_axes);
    OperatorConf unsorted_segment_sum_like_op;
    unsorted_segment_sum_like_op.set_name("System-AutoGrad-" + op.op_name());
    UnsortedSegmentSumLikeOpConf* conf =
        unsorted_segment_sum_like_op.mutable_unsorted_segment_sum_like_conf();
    conf->set_axis(axis);
    conf->set_segment_ids(GenLogicalBlobName(op.BnInOp2Lbi("indices")));
    conf->set_data(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    conf->set_like(GenLogicalBlobName(op.BnInOp2Lbi("in")));
    conf->set_out("out");
    op_confs->push_back(unsorted_segment_sum_like_op);
    DiffLbi4BnInOp("in")->set_op_name(unsorted_segment_sum_like_op.name());
    DiffLbi4BnInOp("in")->set_blob_name(conf->out());
  }
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kGatherConf, &GenerateBackwardOpConf);

}  // namespace oneflow
