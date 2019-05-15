#include "oneflow/core/job_completer/autovar.h"

namespace oneflow {

namespace {

void GenerateInputVarOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<const BlobDesc&(const std::string&)>& BlobDesc4ModelBn) {
  CHECK(op.op_conf().has_fully_connected_conf());
  OperatorConf fully_connected_conf(op.op_conf());
  auto* mut_conf = fully_connected_conf.mutable_fully_connected_conf();
  const auto& conf = op.op_conf().fully_connected_conf();
  if (!conf.has_weight()) {
    OperatorConf weight_var_op =
        GenerateVariableOpConf(BlobDesc4ModelBn("weight"), op.op_name() + "-weight", "weight");
    if (conf.has_weight_initializer()) {
      *(weight_var_op.mutable_variable_conf()->mutable_initializer()) = conf.weight_initializer();
    }
    op_confs->push_back(weight_var_op);
    mut_conf->set_weight(weight_var_op.name() + "/out");
  }
  if (conf.use_bias()) {
    if (!conf.has_bias()) {
      OperatorConf bias_var_op =
          GenerateVariableOpConf(BlobDesc4ModelBn("bias"), op.op_name() + "-bias", "bias");
      if (conf.has_bias_initializer()) {
        *(bias_var_op.mutable_variable_conf()->mutable_initializer()) = conf.bias_initializer();
      }
      op_confs->push_back(bias_var_op);
      mut_conf->set_bias(bias_var_op.name() + "/out");
    }
  }
  op_confs->push_back(fully_connected_conf);
}

}  // namespace

REGISTER_OP_INPUT_VAR(OperatorConf::kFullyConnectedConf, &GenerateInputVarOpConf);

}  // namespace oneflow
