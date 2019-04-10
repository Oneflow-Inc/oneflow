#include "oneflow/core/job_completer/autovar.h"

namespace oneflow {

namespace {

void GenerateInputVarOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<const BlobDesc&(const std::string&)>& UnitBatchSizeBlobDesc4BnInOp) {
  CHECK(op.op_conf().has_layer_norm_conf());
  OperatorConf layer_norm_op_conf(op.op_conf());
  auto* mut_conf = layer_norm_op_conf.mutable_layer_norm_conf();
  const auto& conf = op.op_conf().layer_norm_conf();
  if (conf.center()) {
    if (!conf.has_beta()) {
      OperatorConf beta_var_op = GenerateVariableOpConf(UnitBatchSizeBlobDesc4BnInOp("beta"),
                                                        op.op_name() + "-beta", "beta");
      beta_var_op.mutable_variable_conf()
          ->mutable_initializer()
          ->mutable_constant_conf()
          ->set_value(0.0);
      op_confs->push_back(beta_var_op);
      mut_conf->set_beta(beta_var_op.name() + "/out");
    }
  }
  if (conf.scale()) {
    if (!conf.has_gamma()) {
      OperatorConf gamma_var_op = GenerateVariableOpConf(UnitBatchSizeBlobDesc4BnInOp("gamma"),
                                                         op.op_name() + "-gamma", "gamma");
      gamma_var_op.mutable_variable_conf()
          ->mutable_initializer()
          ->mutable_constant_conf()
          ->set_value(1.0);
      op_confs->push_back(gamma_var_op);
      mut_conf->set_gamma(gamma_var_op.name() + "/out");
    }
  }
  op_confs->push_back(layer_norm_op_conf);
}

}  // namespace

REGISTER_OP_INPUT_VAR(OperatorConf::kLayerNormConf, &GenerateInputVarOpConf);

}  // namespace oneflow
