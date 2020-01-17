#include "oneflow/core/job_completer/optimizer.h"

namespace oneflow {

void GenerateOptimizerOpConfWrapperStruct::Call(const VariableOp& var_op,
                                                const ParallelConf& parallel_conf,
                                                JobBuilder* job_builder,
                                                const LogicalBlobId& diff_lbi_of_var_out) const {
  (*func_)(var_op, parallel_conf, job_builder, diff_lbi_of_var_out);
}

void GenerateOptimizerOpConfIf(const VariableOp& var_op, const ParallelConf& parallel_conf,
                               JobBuilder* job_builder, const LogicalBlobId& diff_lbi_of_var_out) {
  const auto& train_conf = GlobalJobDesc().job_conf().train_conf();
  auto optimizer_case = train_conf.model_update_conf().normal_mdupdt_case();
  auto* obj = NewObj<GenerateOptimizerOpConfWrapperStruct>(optimizer_case);
  obj->Call(var_op, parallel_conf, job_builder, diff_lbi_of_var_out);
}

void AddOptimizerOpConf(const OpGraph& op_graph, JobBuilder* job_builder,
                        const HashMap<LogicalBlobId, LogicalBlobId>& lbi2diff_lbi) {
  op_graph.ForEachNode([&](OpNode* op_node) {
    const VariableOp* var_op = dynamic_cast<const VariableOp*>(&op_node->op());
    if (var_op == nullptr) { return; }
    if (lbi2diff_lbi.find(var_op->BnInOp2Lbi(var_op->SoleObn())) == lbi2diff_lbi.end()) { return; }

    LogicalBlobId diff_lbi_of_var_out = lbi2diff_lbi.at(var_op->BnInOp2Lbi(var_op->SoleObn()));
    const auto& parallel_desc = op_node->parallel_desc();
    GenerateOptimizerOpConfIf(*var_op, parallel_desc.parallel_conf(), job_builder,
                              diff_lbi_of_var_out);
  });
}

template<typename T>
void ConstructMdUpdtOpConf(const VariableOp& op, const LogicalBlobId& diff_lbi_of_var_out,
                           JobBuilder* job_builder, T* mdupdt_op_conf) {
  const auto& train_conf = job_builder->job().job_conf().train_conf();
  *mdupdt_op_conf->mutable_user_conf() = train_conf.model_update_conf();
  mdupdt_op_conf->set_model_diff(GenLogicalBlobName(diff_lbi_of_var_out));
  mdupdt_op_conf->set_model(GenLogicalBlobName(op.BnInOp2Lbi("out")));
  mdupdt_op_conf->set_train_step(train_conf.train_step_lbn());
  const std::string& primary_lr_lbn = train_conf.primary_lr_lbn();
  const std::string& secondary_lr_lbn = train_conf.secondary_lr_lbn();
  if (op.op_conf().variable_conf().model_name() == "weight") {
    mdupdt_op_conf->set_learning_rate(primary_lr_lbn);
    mdupdt_op_conf->set_l1(train_conf.weight_l1());
    mdupdt_op_conf->set_l2(train_conf.weight_l2());
  } else if (op.op_conf().variable_conf().model_name() == "bias") {
    mdupdt_op_conf->set_learning_rate(secondary_lr_lbn);
    mdupdt_op_conf->set_l1(train_conf.bias_l1());
    mdupdt_op_conf->set_l2(train_conf.bias_l2());
  } else {
    mdupdt_op_conf->set_learning_rate(primary_lr_lbn);
    mdupdt_op_conf->set_l1(0);
    mdupdt_op_conf->set_l2(0);
  }
}

#define INSTANTIATE_CONSTRUCTOR_MDUPDT_OP_CONF(T)                                  \
  template void ConstructMdUpdtOpConf<T>(const VariableOp& op,                     \
                                         const LogicalBlobId& diff_lbi_of_var_out, \
                                         JobBuilder* job_builder, T* mdupdt_op_conf)

INSTANTIATE_CONSTRUCTOR_MDUPDT_OP_CONF(NaiveModelUpdateOpConf);
INSTANTIATE_CONSTRUCTOR_MDUPDT_OP_CONF(MomentumModelUpdateOpConf);
INSTANTIATE_CONSTRUCTOR_MDUPDT_OP_CONF(RMSPropModelUpdateOpConf);
INSTANTIATE_CONSTRUCTOR_MDUPDT_OP_CONF(LARSModelUpdateOpConf);
INSTANTIATE_CONSTRUCTOR_MDUPDT_OP_CONF(AdamModelUpdateOpConf);
INSTANTIATE_CONSTRUCTOR_MDUPDT_OP_CONF(LazyAdamModelUpdateOpConf);

}  // namespace oneflow
