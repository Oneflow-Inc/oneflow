#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

Maybe<void> GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) {
  CHECK_OR_RETURN(op.op_conf().has_scalar_mul_by_tensor_conf());
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf scalar_mul_by_tensor_grad_op;
    scalar_mul_by_tensor_grad_op.set_name(op.op_name() + "_grad");
    ScalarMulByTensorOpConf* conf =
        scalar_mul_by_tensor_grad_op.mutable_scalar_mul_by_tensor_conf();
    conf->set_in(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    conf->set_scalar(GenLogicalBlobName(op.BnInOp2Lbi("scalar")));
    conf->set_out("out");
    op_confs->push_back(scalar_mul_by_tensor_grad_op);
    DiffLbi4BnInOp("in")->set_op_name(scalar_mul_by_tensor_grad_op.name());
    DiffLbi4BnInOp("in")->set_blob_name(conf->out());
  }
  if (DiffLbi4BnInOp("scalar") != nullptr) {
    OperatorConf multiply_op;
    multiply_op.set_name(op.op_name() + "_grad_multiply");
    MultiplyOpConf* multiply_conf = multiply_op.mutable_multiply_conf();
    multiply_conf->set_out("out");
    multiply_conf->set_in_0(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    multiply_conf->set_in_1(GenLogicalBlobName(op.BnInOp2Lbi("in")));
    op_confs->push_back(multiply_op);
    OperatorConf reduce_sum_op;
    reduce_sum_op.set_name(op.op_name() + "_grad_reduce_sum");
    ReduceSumOpConf* reduce_sum_conf = reduce_sum_op.mutable_reduce_sum_conf();
    reduce_sum_conf->set_in(GenLogicalBlobName(multiply_op.name(), multiply_conf->out()));
    reduce_sum_conf->set_out("out");
    int64_t num_axes = LogicalBlobDesc4BnInOp("out").shape().NumAxes();
    FOR_RANGE(int64_t, i, 0, num_axes) { reduce_sum_conf->add_axis(i); }
    op_confs->push_back(reduce_sum_op);
    DiffLbi4BnInOp("scalar")->set_op_name(reduce_sum_op.name());
    DiffLbi4BnInOp("scalar")->set_blob_name(reduce_sum_conf->out());
  }
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kScalarMulByTensorConf, &GenerateBackwardOpConf);

}  // namespace oneflow
