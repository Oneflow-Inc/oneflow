#include "oneflow/core/autograd/autovar.h"

namespace oneflow {

namespace {

void GenerateInputVarOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) {
  CHECK(op.op_conf().has_layer_norm_conf());
  OperatorConf layer_norm_op_conf(op.op_conf());
  auto* mut_conf = layer_norm_op_conf.mutable_layer_norm_conf();
  const auto& conf = op.op_conf().layer_norm_conf();
  if (conf.center()) {
    if (!conf.has_beta()) {
      const OperatorConf& beta_var_op =
          GenerateVariableOpConf(LogicalBlobDesc4BnInOp("beta"), op.op_name() + "-beta", "beta");
      op_confs->push_back(beta_var_op);
      mut_conf->set_beta(beta_var_op.name() + "/out");
    }
  }
  if (conf.scale()) {
    if (!conf.has_gamma()) {
      const OperatorConf& gamma_var_op =
          GenerateVariableOpConf(LogicalBlobDesc4BnInOp("gamma"), op.op_name() + "-gamma", "gamma");
      op_confs->push_back(gamma_var_op);
      mut_conf->set_gamma(gamma_var_op.name() + "/out");
    }
  }
  op_confs->push_back(layer_norm_op_conf);
}

}  // namespace

REGISTER_OP_INPUT_VAR(OperatorConf::kLayerNormConf, &GenerateInputVarOpConf);

}  // namespace oneflow
