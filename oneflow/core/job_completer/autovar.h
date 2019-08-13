#ifndef ONEFLOW_CORE_JOB_COMPLETER_AUTOVAR_H_
#define ONEFLOW_CORE_JOB_COMPLETER_AUTOVAR_H_

#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/op_graph.h"

namespace oneflow {

void AutoVar(const OpGraph& op_graph, JobBuilder* job_builder);

OperatorConf GenerateVariableOpConf(const BlobDesc& blob_desc, const std::string& name,
                                    const std::string& model_name);

class GenerateInputVarOpConfWrapperStruct final {
 public:
  using Func = std::function<void(const Operator&, std::vector<OperatorConf>*,
                                  const std::function<const BlobDesc&(const std::string&)>&)>;
  GenerateInputVarOpConfWrapperStruct(const Func& f) : func_(std::make_unique<Func>(f)) {}
  void Call(
      const Operator& op, std::vector<OperatorConf>* op_confs,
      const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) const {
    (*func_)(op, op_confs, LogicalBlobDesc4BnInOp);
  }

 private:
  const std::unique_ptr<const Func> func_;
};

#define REGISTER_OP_INPUT_VAR(op_type_case, gen_grad_func)                  \
  REGISTER_CLASS_CREATOR(op_type_case, GenerateInputVarOpConfWrapperStruct, \
                         ([] { return new GenerateInputVarOpConfWrapperStruct(gen_grad_func); }))

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COMPLETER_AUTOVAR_H_
