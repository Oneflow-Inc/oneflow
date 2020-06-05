#include "oneflow/core/job_completer/optimizer_registry.h"

namespace oneflow {
namespace {

HashMap<std::string, OptimizerBase*>* MutOptimizerRegistry() {
  static HashMap<std::string, OptimizerBase*> registry;
  return &registry;
}

}  // namespace

OptimizerBase* OptimizerRegistry::Lookup(const std::string& name) {
  if (MutOptimizerRegistry()->empty()) { LOG(FATAL) << "no optimizer registered"; }
  const auto it = MutOptimizerRegistry()->find(name);
  if (it == MutOptimizerRegistry()->end()) {
    std::string all_registered_name;
    for (const auto tuple : *MutOptimizerRegistry()) {
      all_registered_name += ", ";
      all_registered_name += tuple.first;
    }
    LOG(FATAL) << "optimizer " << name << " not found, all: " << all_registered_name;
  }
  CHECK(it->second != nullptr);
  return it->second;
}

Maybe<void> OptimizerRegistry::LookupAndBuild(const std::string& name, const VariableOp& var_op,
                                              const ParallelConf& parallel_conf,
                                              const LogicalBlobId& diff_lbi_of_var_out,
                                              const ::oneflow::TrainConf& train_conf) {
  const std::string var_op_conf_txt = PbMessage2TxtString(var_op.op_conf());
  const std::string parallel_conf_txt = PbMessage2TxtString(parallel_conf);
  const std::string diff_lbi_of_var_out_txt = PbMessage2TxtString(diff_lbi_of_var_out);
  const std::string train_conf_txt = PbMessage2TxtString(train_conf);
  OptimizerRegistry::Lookup(name)->Build(var_op_conf_txt, parallel_conf_txt,
                                         diff_lbi_of_var_out_txt, train_conf_txt);
  return Maybe<void>::Ok();
}

Maybe<void> OptimizerRegistry::Register(std::string name, OptimizerBase* optimizer) {
  CHECK_OR_RETURN(MutOptimizerRegistry()->find(name) == MutOptimizerRegistry()->end())
      << "optmizer " << name << " registered";
  CHECK(MutOptimizerRegistry()->emplace(name, optimizer).second);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
