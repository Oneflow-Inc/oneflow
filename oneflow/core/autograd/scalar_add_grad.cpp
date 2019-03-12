#include "oneflow/core/autograd/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(const Operator& op, std::vector<OperatorConf>* op_confs,
                            const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
                            const std::function<DataType(const std::string&)>& DataType4BnInOp) {
  CHECK(op.op_conf().has_scalar_add_conf());
  if (DiffLbi4BnInOp("in") != nullptr) { *DiffLbi4BnInOp("in") = *DiffLbi4BnInOp("out"); }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kScalarAddConf, &GenerateBackwardOpConf);

}  // namespace oneflow
