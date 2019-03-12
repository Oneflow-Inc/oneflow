#include "oneflow/core/autograd/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(const Operator& op, std::vector<OperatorConf>* op_confs,
                            const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
                            const std::function<DataType(const std::string&)>& DataType4BnInOp) {
  CHECK(op.op_conf().has_scalar_add_conf());
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf clone_op;
    clone_op.set_name(op.op_name() + "_grad");
    CloneOpConf* clone_op_conf = clone_op.mutable_clone_conf();
    clone_op_conf->set_out_num(1);
    op_confs->push_back(clone_op);
    DiffLbi4BnInOp("in")->set_op_name(clone_op.name());
    DiffLbi4BnInOp("in")->set_blob_name("out_0");
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kScalarAddConf, &GenerateBackwardOpConf);

}  // namespace oneflow
