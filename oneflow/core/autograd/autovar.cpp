#include "oneflow/core/autograd/autovar.h"
#include "oneflow/core/job/job_conf_builder.h"

namespace oneflow {

OperatorConf GenerateVariableOpConf(const BlobDesc& blob_desc, const std::string& name,
                                    const std::string& model_name) {
  OperatorConf var_op;
  var_op.set_name(name);
  VariableOpConf* var_op_conf = var_op.mutable_variable_conf();
  var_op_conf->set_out("out");
  blob_desc.shape().ToProto(var_op_conf->mutable_shape());
  var_op_conf->set_data_type(blob_desc.data_type());
  var_op_conf->set_model_name(model_name);
  return var_op;
}

void GenerateInputVarOpConfIf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) {
  if (IsClassRegistered<GenerateInputVarOpConfWrapperStruct>(op.op_conf().op_type_case())) {
    auto* obj = NewObj<GenerateInputVarOpConfWrapperStruct>(op.op_conf().op_type_case());
    obj->Call(op, op_confs, LogicalBlobDesc4BnInOp);
  }
}

void AutoVar(const OpGraph& op_graph, JobConf1* job_conf) {
  JobConfBuilder job_conf_builder(job_conf);
  op_graph.ForEachNode([&](OpNode* op_node) {
    std::vector<OperatorConf> ops;
    auto LogicalBlobDesc4BnInOp = [&](const std::string& bn) -> const BlobDesc& {
      return op_graph.GetLogicalBlobDesc(op_node->op().BnInOp2Lbi(bn));
    };
    GenerateInputVarOpConfIf(op_node->op(), &ops, LogicalBlobDesc4BnInOp);
    if (!ops.empty()) {
      job_conf_builder.AddOrMutOps(op_node->parallel_desc().parallel_conf(), ops);
    }
  });
}

}  // namespace oneflow
