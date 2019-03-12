#include "oneflow/core/autograd/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_add_conf());
  const AddOpConf& conf = op.op_conf().add_conf();
  int32_t out_num = 0;
  FOR_RANGE(int32_t, i, 0, conf.in_size()) {
    if (DiffLbi4BnInOp("in_" + std::to_string(i)) != nullptr) { out_num++; }
  }
  if (out_num > 0) {
    OperatorConf clone_op;
    clone_op.set_name(op.op_name() + "_grad");
    CloneOpConf* clone_op_conf = clone_op.mutable_clone_conf();
    clone_op_conf->set_out_num(out_num);
    int32_t out_count = 0;
    FOR_RANGE(int32_t, i, 0, conf.in_size()) {
      const std::string ibn = "in_" + std::to_string(i);
      if (DiffLbi4BnInOp(ibn) != nullptr) {
        DiffLbi4BnInOp(ibn)->set_blob_name("out_" + std::to_string(out_count));
        out_count = out_count + 1;
      }
    }
    CHECK_EQ(out_count, out_num);
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kAddConf, &GenerateBackwardOpConf);

}  // namespace oneflow
