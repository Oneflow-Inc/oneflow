#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) {
  CHECK(op.op_conf().has_stack_conf());
  const auto& stack_conf = op.op_conf().stack_conf();
  OperatorConf op_conf;
  op_conf.set_name(op.op_conf().name() + "_grad");
  auto* stack_grad_conf = op_conf.mutable_stack_grad_conf();
  stack_grad_conf->set_in(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
  stack_grad_conf->set_axis(stack_conf.axis());
  FOR_RANGE(int32_t, i, 0, stack_conf.in_size()) {
    const std::string& ibn_of_stack_op = op.input_bns().Get(i);
    const std::string& obn = "out_" + std::to_string(i);
    stack_grad_conf->add_like(GenLogicalBlobName(op.BnInOp2Lbi(ibn_of_stack_op)));
    stack_grad_conf->add_out(obn);
    if (DiffLbi4BnInOp(ibn_of_stack_op) != nullptr) {
      DiffLbi4BnInOp(ibn_of_stack_op)->set_op_name(op_conf.name());
      DiffLbi4BnInOp(ibn_of_stack_op)->set_blob_name(obn);
    }
  }
  op_confs->push_back(op_conf);
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kStackConf, &GenerateBackwardOpConf);

}  // namespace oneflow
