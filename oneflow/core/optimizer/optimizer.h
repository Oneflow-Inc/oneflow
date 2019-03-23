#ifndef ONEFLOW_CORE_OPTIMIZER_OPTIMIZER_H_
#define ONEFLOW_CORE_OPTIMIZER_OPTIMIZER_H_

#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/operator/variable_op.h"

namespace oneflow {

void AddOptimizerOpConf(const OpGraph& op_graph, JobConf1* job_conf,
                        const HashMap<LogicalBlobId, LogicalBlobId>& lbi2diff_lbi,
                        const LogicalBlobId& total_loss_instance_num_lbi);

template<typename T>
void ConstructMdUpdtOpConf(
    const VariableOp& op,
    const std::function<const LogicalBlobId&(const std::string&)>& DiffLbi4BnInOp,
    const LogicalBlobId& total_loss_instance_num_lbi, T*);

void GenerateOptimizerOpConfIf(
    const VariableOp& var_op, std::vector<OperatorConf>* op_confs,
    const std::function<const LogicalBlobId&(const std::string&)>& DiffLbi4BnInOp,
    const LogicalBlobId& total_loss_instance_num_lbi);

class GenerateOptimizerOpConfWrapperStruct final {
 public:
  using Func = std::function<void(const VariableOp&, std::vector<OperatorConf>*,
                                  const std::function<const LogicalBlobId&(const std::string&)>&,
                                  const LogicalBlobId&)>;
  GenerateOptimizerOpConfWrapperStruct(const Func& f) : func_(std::make_unique<Func>(f)) {}
  void Call(const VariableOp& var_op, std::vector<OperatorConf>* op_confs,
            const std::function<const LogicalBlobId&(const std::string&)>& DiffLbi4BnInOp,
            const LogicalBlobId& total_loss_instance_num_lbi) const {
    (*func_)(var_op, op_confs, DiffLbi4BnInOp, total_loss_instance_num_lbi);
  }

 private:
  const std::unique_ptr<const Func> func_;
};

#define REGISTER_OPTIMIZER(model_update_case, gen_grad_func)                      \
  REGISTER_CLASS_CREATOR(model_update_case, GenerateOptimizerOpConfWrapperStruct, \
                         ([] { return new GenerateOptimizerOpConfWrapperStruct(gen_grad_func); }))

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPTIMIZER_OPTIMIZER_H_
