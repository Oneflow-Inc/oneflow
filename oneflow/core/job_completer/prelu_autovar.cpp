#include "oneflow/core/job_completer/autovar.h"

namespace oneflow {

namespace {

void GenerateInputVarOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<const BlobDesc&(const std::string&)>& BlobDesc4ModelBn) {
  CHECK(op.op_conf().has_prelu_conf());
  OperatorConf prelu_conf(op.op_conf());
  auto* mut_conf = prelu_conf.mutable_prelu_conf();
  const auto& conf = op.op_conf().prelu_conf();
  if (!conf.has_alpha()) {
    OperatorConf alpha_var_op =
        GenerateVariableOpConf(BlobDesc4ModelBn("alpha"), op.op_name() + "-alpha", "alpha");
    alpha_var_op.mutable_variable_conf()->mutable_initializer()->mutable_constant_conf()->set_value(
        conf.alpha_init());
    op_confs->push_back(alpha_var_op);
    mut_conf->set_alpha(alpha_var_op.name() + "/out");
  }
  op_confs->push_back(prelu_conf);
}

}  // namespace

REGISTER_OP_INPUT_VAR(OperatorConf::kPreluConf, &GenerateInputVarOpConf);

}  // namespace oneflow
