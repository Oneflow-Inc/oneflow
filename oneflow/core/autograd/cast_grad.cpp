#include "oneflow/core/autograd/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(const Operator& op, std::vector<OperatorConf>* op_confs,
                            const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
                            const std::function<DataType(const std::string&)>& DataType4BnInOp) {
  CHECK(op.op_conf().has_cast_conf());
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf reverse_cast_op;
    reverse_cast_op.set_name(op.op_name() + "_grad");
    CastOpConf* reverse_cast_op_conf = reverse_cast_op.mutable_cast_conf();
    reverse_cast_op_conf->set_out("out");
    reverse_cast_op_conf->set_in(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    reverse_cast_op_conf->set_data_type(DataType4BnInOp("in"));
    op_confs->push_back(reverse_cast_op);
    DiffLbi4BnInOp("in")->set_op_name(reverse_cast_op.name());
    DiffLbi4BnInOp("in")->set_blob_name("out");
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kCastConf, &GenerateBackwardOpConf);

}  // namespace oneflow
