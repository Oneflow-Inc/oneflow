#include "oneflow/core/job_completer/optimizer.h"

namespace oneflow {

void GenerateOptimizerOpConfIf(
    const VariableOp& var_op, std::vector<OperatorConf>* op_confs,
    const std::function<const LogicalBlobId&(const std::string&)>& DiffLbi4BnInOp,
    const LogicalBlobId& total_loss_instance_num_lbi) {
  const auto& train_conf =
      Global<JobDesc>::Get()->other_conf().predict_conf().tmp_split_fw_bw_train_conf();
  auto optimizer_case = train_conf.model_update_conf().normal_mdupdt_case();
  auto* obj = NewObj<GenerateOptimizerOpConfWrapperStruct>(optimizer_case);
  obj->Call(var_op, op_confs, DiffLbi4BnInOp, total_loss_instance_num_lbi);
}

void AddOptimizerOpConf(const OpGraph& op_graph, Job* job,
                        const HashMap<LogicalBlobId, LogicalBlobId>& lbi2diff_lbi,
                        const LogicalBlobId& total_loss_instance_num_lbi) {
  JobBuilder job_builder(job);
  op_graph.ForEachNode([&](OpNode* op_node) {
    const VariableOp* var_op = dynamic_cast<const VariableOp*>(&op_node->op());
    if (var_op == nullptr) { return; }
    if (lbi2diff_lbi.find(var_op->BnInOp2Lbi(var_op->SoleObn())) == lbi2diff_lbi.end()) { return; }
    std::vector<OperatorConf> optimizer_op_confs;
    auto DiffLbi4BnInOp = [&](const std::string& bn) -> const LogicalBlobId& {
      return lbi2diff_lbi.at(var_op->BnInOp2Lbi(bn));
    };
    GenerateOptimizerOpConfIf(*var_op, &optimizer_op_confs, DiffLbi4BnInOp,
                              total_loss_instance_num_lbi);
    job_builder.AddOps(op_node->parallel_desc().parallel_conf(), optimizer_op_confs);
  });
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
