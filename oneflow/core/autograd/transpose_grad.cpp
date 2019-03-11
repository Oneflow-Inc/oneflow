#include "oneflow/core/autograd/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_transpose_conf());
  const TransposeOpConf& conf = op.op_conf().transpose_conf();
  const PbRf<int32_t>& perm = conf.perm();
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf transpose_op;
    transpose_op.set_name(op.op_name() + "_grad");
    TransposeOpConf* transpose_op_conf = transpose_op.mutable_transpose_conf();
    transpose_op_conf->set_out("out");
    transpose_op_conf->set_in(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    PbRf<int32_t>* invert_perm = transpose_op_conf->mutable_perm();
    invert_perm->Reserve(perm.size());
    invert_perm->CopyFrom(perm);
    FOR_RANGE(size_t, i, 0, perm.size()) { (*invert_perm)[perm[i]] = i; }
    op_confs->push_back(transpose_op);
    DiffLbi4BnInOp("in")->set_op_name(transpose_op.name());
    DiffLbi4BnInOp("in")->set_blob_name("out");
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kTransposeConf, &GenerateBackwardOpConf);

}  // namespace oneflow
