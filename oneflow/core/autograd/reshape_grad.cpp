#include "oneflow/core/autograd/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(const Operator& op, std::vector<OperatorConf>* op_confs,
                            const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
                            const std::function<DataType(const std::string&)>& DataType4BnInOp) {
  CHECK(op.op_conf().has_reshape_conf());
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf reverse_reshape_op;
    reverse_reshape_op.set_name(op.op_name() + "_grad");
    ReshapeLikeOpConf* reshape_like_op_conf = reverse_reshape_op.mutable_reshape_like_conf();
    reshape_like_op_conf->set_x(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    reshape_like_op_conf->set_like(GenLogicalBlobName(op.BnInOp2Lbi("in")));
    reshape_like_op_conf->set_y("y");
    op_confs->push_back(reverse_reshape_op);
    DiffLbi4BnInOp("in")->set_op_name(reverse_reshape_op.name());
    DiffLbi4BnInOp("in")->set_blob_name("y");
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kReshapeConf, &GenerateBackwardOpConf);

}  // namespace oneflow
