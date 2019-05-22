#include "oneflow/core/job_completer/optimizer.h"

namespace oneflow {

void GenerateOptimizerOpConfWrapperStruct::Call(
    const VariableOp& var_op, const ParallelConf& parallel_conf, JobBuilder* job_builder,
    const std::function<const LogicalBlobId&(const std::string&)>& DiffLbi4BnInOp,
    const LogicalBlobId& total_loss_instance_num_lbi) const {
  (*func_)(var_op, parallel_conf, job_builder, DiffLbi4BnInOp, total_loss_instance_num_lbi);
}

void GenerateOptimizerOpConfIf(
    const VariableOp& var_op, const ParallelConf& parallel_conf, JobBuilder* job_builder,
    const std::function<const LogicalBlobId&(const std::string&)>& DiffLbi4BnInOp,
    const LogicalBlobId& total_loss_instance_num_lbi) {
  const auto& train_conf =
      Global<JobDesc>::Get()->other_conf().predict_conf().tmp_split_fw_bw_train_conf();
  auto optimizer_case = train_conf.model_update_conf().normal_mdupdt_case();
  std::unique_ptr<GenerateOptimizerOpConfWrapperStruct> obj;
  obj.reset(NewObj<GenerateOptimizerOpConfWrapperStruct>(optimizer_case));
  obj->Call(var_op, parallel_conf, job_builder, DiffLbi4BnInOp, total_loss_instance_num_lbi);
}

void AddOptimizerOpConf(
    const OpGraph& op_graph, Job* job, const HashMap<LogicalBlobId, LogicalBlobId>& lbi2diff_lbi,
    const std::function<const LogicalBlobId&(const ParallelDesc&)>& LossInstanceNum4ParallelDesc) {
  JobBuilder job_builder(job);
  op_graph.ForEachNode([&](OpNode* op_node) {
    const VariableOp* var_op = dynamic_cast<const VariableOp*>(&op_node->op());
    if (var_op == nullptr) { return; }
    if (lbi2diff_lbi.find(var_op->BnInOp2Lbi(var_op->SoleObn())) == lbi2diff_lbi.end()) { return; }
    std::vector<OperatorConf> optimizer_op_confs;
    auto DiffLbi4BnInOp = [&](const std::string& bn) -> const LogicalBlobId& {
      return lbi2diff_lbi.at(var_op->BnInOp2Lbi(bn));
    };
    const auto& parallel_desc = op_node->parallel_desc();
    GenerateOptimizerOpConfIf(*var_op, op_node->parallel_desc().parallel_conf(), &job_builder,
                              DiffLbi4BnInOp, LossInstanceNum4ParallelDesc(parallel_desc));
    job_builder.AddOps(parallel_desc.parallel_conf(), optimizer_op_confs);
  });
}

void BindTwoVariableOpObnSbpConf(const std::string& lhs_op_name, const std::string& rhs_op_name,
                                 JobBuilder* job_builder) {
  job_builder->BindIdenticalSbpOpBlobArgPair(GenOpBlobArg(lhs_op_name, "out"),
                                             GenOpBlobArg(rhs_op_name, "out"));
}

template<typename T>
void ConstructMdUpdtOpConf(
    const VariableOp& op,
    const std::function<const LogicalBlobId&(const std::string&)>& DiffLbi4BnInOp,
    const LogicalBlobId& total_loss_instance_num_lbi, T* mdupdt_op_conf) {
  const auto& train_conf =
      Global<JobDesc>::Get()->other_conf().predict_conf().tmp_split_fw_bw_train_conf();
  *mdupdt_op_conf->mutable_user_conf() = train_conf.model_update_conf();
  mdupdt_op_conf->set_model_diff(GenLogicalBlobName(DiffLbi4BnInOp("out")));
  mdupdt_op_conf->set_total_instance_num_diff(GenLogicalBlobName(total_loss_instance_num_lbi));
  mdupdt_op_conf->set_model(GenLogicalBlobName(op.BnInOp2Lbi("out")));
  float primary_lr = Global<JobDesc>::Get()->primary_lr();
  float secondary_lr = Global<JobDesc>::Get()->secondary_lr();
  if (secondary_lr < 0) { secondary_lr = primary_lr; }
  if (op.op_conf().variable_conf().model_name() == "weight") {
    mdupdt_op_conf->set_learning_rate(primary_lr);
    mdupdt_op_conf->set_l1(Global<JobDesc>::Get()->weight_l1());
    mdupdt_op_conf->set_l2(Global<JobDesc>::Get()->weight_l2());
  } else if (op.op_conf().variable_conf().model_name() == "bias") {
    mdupdt_op_conf->set_learning_rate(secondary_lr);
    mdupdt_op_conf->set_l1(Global<JobDesc>::Get()->bias_l1());
    mdupdt_op_conf->set_l2(Global<JobDesc>::Get()->bias_l2());
  } else {
    mdupdt_op_conf->set_learning_rate(primary_lr);
    mdupdt_op_conf->set_l1(0);
    mdupdt_op_conf->set_l2(0);
  }
}

#define INSTANTIATE_CONSTRUCTOR_MDUPDT_OP_CONF(T)                                    \
  template void ConstructMdUpdtOpConf<T>(                                            \
      const VariableOp& op,                                                          \
      const std::function<const LogicalBlobId&(const std::string&)>& DiffLbi4BnInOp, \
      const LogicalBlobId& total_loss_instance_num_lbi, T* mdupdt_op_conf)

INSTANTIATE_CONSTRUCTOR_MDUPDT_OP_CONF(NaiveModelUpdateOpConf);
INSTANTIATE_CONSTRUCTOR_MDUPDT_OP_CONF(MomentumModelUpdateOpConf);
INSTANTIATE_CONSTRUCTOR_MDUPDT_OP_CONF(RMSPropModelUpdateOpConf);
INSTANTIATE_CONSTRUCTOR_MDUPDT_OP_CONF(LARSModelUpdateOpConf);
INSTANTIATE_CONSTRUCTOR_MDUPDT_OP_CONF(AdamModelUpdateOpConf);

}  // namespace oneflow
