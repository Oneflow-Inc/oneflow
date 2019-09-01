#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) {
  CHECK(op.op_conf().has_bias_add_conf());
  if (DiffLbi4BnInOp("a") != nullptr) { *DiffLbi4BnInOp("a") = *DiffLbi4BnInOp("out"); }
  if (DiffLbi4BnInOp("b") != nullptr) {
    OperatorConf reduce_sum_op;
    reduce_sum_op.set_name(op.op_name() + "_grad_b");
    ReduceSumOpConf* reduce_sum_op_conf = reduce_sum_op.mutable_reduce_sum_conf();
    reduce_sum_op_conf->set_in(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    reduce_sum_op_conf->set_out("out");

    const int32_t bias_add_axis = op.op_conf().bias_add_conf().axis();
    const int32_t num_axes = LogicalBlobDesc4BnInOp("a").shape().NumAxes();
    FOR_RANGE(int32_t, i, 0, num_axes) {
      if (i != bias_add_axis) { reduce_sum_op_conf->add_axis(i); }
    }
    reduce_sum_op_conf->set_keep_dims(false);
    op_confs->push_back(reduce_sum_op);
    DiffLbi4BnInOp("b")->set_op_name(reduce_sum_op.name());
    DiffLbi4BnInOp("b")->set_blob_name("out");
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kBiasAddConf, &GenerateBackwardOpConf);

}  // namespace oneflow
