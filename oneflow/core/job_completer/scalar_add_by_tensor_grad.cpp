#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_scalar_add_by_tensor_conf());
  if (DiffLbi4BnInOp("in") != nullptr) { *DiffLbi4BnInOp("in") = *DiffLbi4BnInOp("out"); }
  if (DiffLbi4BnInOp("scalar") != nullptr) {
    OperatorConf reduce_sum_op;
    reduce_sum_op.set_name(op.op_name() + "_scalar_grad");
    ReduceSumOpConf* reduce_sum_conf = reduce_sum_op.mutable_reduce_sum_conf();
    reduce_sum_conf->set_in(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    reduce_sum_conf->set_out("out");
    op_confs->push_back(reduce_sum_op);
    DiffLbi4BnInOp("scalar")->set_op_name(reduce_sum_op.name());
    DiffLbi4BnInOp("scalar")->set_blob_name(reduce_sum_conf->out());
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kScalarAddByTensorConf, &GenerateBackwardOpConf);

}  // namespace oneflow
