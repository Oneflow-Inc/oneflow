#ifndef ONEFLOW_CORE_JOB_COMPLETER_OPTIMIZER_H_
#define ONEFLOW_CORE_JOB_COMPLETER_OPTIMIZER_H_

#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/operator/variable_op.h"

namespace oneflow {

void AddOptimizerOpConf(const OpGraph& op_graph, JobBuilder* job_builder,
                        const HashMap<LogicalBlobId, LogicalBlobId>& lbi2diff_lbi);

template<typename T>
void ConstructMdUpdtOpConf(const VariableOp& op, const LogicalBlobId& diff_lbi_of_var_out,
                           JobBuilder* job_builder, T*);

class GenerateOptimizerOpConfWrapperStruct final {
 public:
  using Func = std::function<void(const VariableOp&, const ParallelConf&, JobBuilder*,
                                  const LogicalBlobId&)>;
  GenerateOptimizerOpConfWrapperStruct(const Func& f) : func_(std::make_unique<Func>(f)) {}
  void Call(const VariableOp& op, const ParallelConf& parallel_conf, JobBuilder* job_builder,
            const LogicalBlobId& diff_lbi_of_var_out) const;

 private:
  const std::unique_ptr<const Func> func_;
};

#define REGISTER_OPTIMIZER(model_update_case, gen_grad_func)                      \
  REGISTER_CLASS_CREATOR(model_update_case, GenerateOptimizerOpConfWrapperStruct, \
                         ([] { return new GenerateOptimizerOpConfWrapperStruct(gen_grad_func); }))

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COMPLETER_OPTIMIZER_H_
