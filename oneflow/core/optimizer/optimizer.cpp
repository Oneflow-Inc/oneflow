#include "oneflow/core/optimizer/optimizer.h"

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

void AddOptimizerOpConf(const OpGraph& op_graph, JobConf1* job_conf,
                        const HashMap<LogicalBlobId, LogicalBlobId>& lbi2diff_lbi,
                        const LogicalBlobId& total_loss_instance_num_lbi) {
  op_graph.ForEachNode([&](OpNode* op_node) {
    const VariableOp* var_op = dynamic_cast<const VariableOp*>(&op_node->op());
    if (var_op == nullptr) { return; }
    std::vector<OperatorConf> optimizer_op_confs;
    auto DiffLbi4BnInOp = [&](const std::string& bn) -> const LogicalBlobId& {
      return lbi2diff_lbi.at(var_op->BnInOp2Lbi(bn));
    };
    GenerateOptimizerOpConfIf(*var_op, &optimizer_op_confs, DiffLbi4BnInOp,
                              total_loss_instance_num_lbi);
    JobConfBuilder(job_conf).AddOps(op_node->parallel_desc().parallel_conf(), optimizer_op_confs);
  });
}

OperatorConf ConstructMdUpdtOpConf(
    const VariableOp& op,
    const std::function<const LogicalBlobId&(const std::string&)>& DiffLbi4BnInOp,
    const LogicalBlobId& total_loss_instance_num_lbi) {
  const auto& train_conf =
      Global<JobDesc>::Get()->other_conf().predict_conf().tmp_split_fw_bw_train_conf();
  OperatorConf mdupdt_op;
  mdupdt_op.set_name(op.op_name() + "_optimizer");
  auto* mdupdt_op_conf = mdupdt_op.mutable_normal_mdupdt_conf();
  mdupdt_op_conf->set_model_diff(GenLogicalBlobName(DiffLbi4BnInOp("out")));
  mdupdt_op_conf->set_total_instance_num_diff(GenLogicalBlobName(total_loss_instance_num_lbi));
  mdupdt_op_conf->set_model(GenLogicalBlobName(op.BnInOp2Lbi("out")));
  *(mdupdt_op_conf->mutable_user_conf()) = train_conf.model_update_conf();
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
  return mdupdt_op;
}

}  // namespace oneflow
