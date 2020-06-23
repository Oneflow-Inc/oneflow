#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

Maybe<void> GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) {
  CHECK_OR_RETURN(op.op_conf().has_scalar_sub_by_tensor_conf());
  if (DiffLbi4BnInOp("in") != nullptr) { *DiffLbi4BnInOp("in") = *DiffLbi4BnInOp("out"); }
  if (DiffLbi4BnInOp("scalar") != nullptr) {
    OperatorConf reduce_sum_op;
    reduce_sum_op.set_name(op.op_name() + "_grad_reduce_sum");
    ReduceSumOpConf* reduce_sum_conf = reduce_sum_op.mutable_reduce_sum_conf();
    reduce_sum_conf->set_in(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    reduce_sum_conf->set_out("out");
    int64_t num_axes = LogicalBlobDesc4BnInOp("out").shape().NumAxes();
    FOR_RANGE(int64_t, i, 0, num_axes) { reduce_sum_conf->add_axis(i); }
    op_confs->push_back(reduce_sum_op);
    OperatorConf scalar_mul_op;
    scalar_mul_op.set_name(op.op_name() + "_grad_scalar_mul");
    ScalarMulOpConf* scalar_mul_op_conf = scalar_mul_op.mutable_scalar_mul_conf();
    scalar_mul_op_conf->set_in(GenLogicalBlobName(reduce_sum_op.name(), reduce_sum_conf->out()));
    scalar_mul_op_conf->set_float_operand(-1);
    scalar_mul_op_conf->set_out("out");
    op_confs->push_back(scalar_mul_op);
    DiffLbi4BnInOp("scalar")->set_op_name(scalar_mul_op.name());
    DiffLbi4BnInOp("scalar")->set_blob_name(scalar_mul_op_conf->out());
  }
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kScalarSubByTensorConf, &GenerateBackwardOpConf);

}  // namespace oneflow
