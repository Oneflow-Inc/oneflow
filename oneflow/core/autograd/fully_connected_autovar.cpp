#include "oneflow/core/autograd/autovar.h"

namespace oneflow {

namespace {

void GenerateInputVarOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<const BlobDesc&(const std::string&)>& UnitBatchSizeBlobDesc4BnInOp,
    const LogicalBlobId& tick_lbi) {
  CHECK(op.op_conf().has_fully_connected_conf());
  OperatorConf fully_connected_conf(op.op_conf());
  auto* mut_conf = fully_connected_conf.mutable_fully_connected_conf();
  const auto& conf = op.op_conf().fully_connected_conf();
  if (!conf.has_weight()) {
    const OperatorConf& weight_var_op = GenerateVariableOpConf(
        tick_lbi, UnitBatchSizeBlobDesc4BnInOp("weight"), op.op_name() + "-weight", "weight");
    op_confs->push_back(weight_var_op);
    mut_conf->set_weight(weight_var_op.name() + "/out");
  }
  if (conf.use_bias()) {
    if (!conf.has_bias()) {
      const OperatorConf& bias_var_op = GenerateVariableOpConf(
          tick_lbi, UnitBatchSizeBlobDesc4BnInOp("bias"), op.op_name() + "-bias", "bias");
      op_confs->push_back(bias_var_op);
      mut_conf->set_bias(bias_var_op.name() + "/out");
    }
  }
  op_confs->push_back(fully_connected_conf);
}

}  // namespace

REGISTER_OP_INPUT_VAR(OperatorConf::kFullyConnectedConf, &GenerateInputVarOpConf);

}  // namespace oneflow
