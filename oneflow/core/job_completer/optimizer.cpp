#include "oneflow/core/job_completer/optimizer.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job_completer/add_schedule_op.h"
#include "oneflow/core/job/job.pb.h"

namespace oneflow {

void GenerateOptimizerOpConfWrapperStruct::Call(
    const VariableOp& var_op, const ParallelConf& parallel_conf, JobBuilder* job_builder,
    const LogicalBlobId& diff_lbi_of_var_out,
    const LogicalBlobId& total_loss_instance_num_lbi) const {
  (*func_)(var_op, parallel_conf, job_builder, diff_lbi_of_var_out, total_loss_instance_num_lbi);
}

void GenerateOptimizerOpConfIf(const VariableOp& var_op, const ParallelConf& parallel_conf,
                               JobBuilder* job_builder, const LogicalBlobId& diff_lbi_of_var_out,
                               const LogicalBlobId& total_loss_instance_num_lbi) {
  const auto& train_conf = GlobalJobDesc().job_conf().train_conf();
  auto optimizer_case =
      var_op.op_conf().variable_conf().has_model_update_conf()
          ? var_op.op_conf().variable_conf().model_update_conf().normal_mdupdt_case()
          : train_conf.model_update_conf().normal_mdupdt_case();
  auto* obj = NewObj<GenerateOptimizerOpConfWrapperStruct>(optimizer_case);
  obj->Call(var_op, parallel_conf, job_builder, diff_lbi_of_var_out, total_loss_instance_num_lbi);
}

void GenerateDownScaleOpConf(const std::string& name, const ParallelConf& parallel_conf,
                             JobBuilder* job_builder, LogicalBlobId* diff_lbi_of_var_out) {
  int32_t loss_scale_factor = GlobalJobDesc().loss_scale_factor();
  if (loss_scale_factor == 1) { return; }
  float down_scale_factor = 1.0 / loss_scale_factor;

  OperatorConf down_scale_mul_op;
  down_scale_mul_op.set_name(name);
  ScalarMulOpConf* conf = down_scale_mul_op.mutable_scalar_mul_conf();
  conf->set_in(GenLogicalBlobName(*diff_lbi_of_var_out));
  conf->set_out("out");
  conf->set_float_operand(down_scale_factor);

  *diff_lbi_of_var_out = GenLogicalBlobId(name + "/out");
  job_builder->AddOps(parallel_conf, {down_scale_mul_op});
}

void AddOptimizerOpConf(
    const OpGraph& op_graph, JobBuilder* job_builder,
    const HashMap<LogicalBlobId, LogicalBlobId>& lbi2diff_lbi,
    const std::function<const LogicalBlobId&(const ParallelDesc&)>& LossInstanceNum4ParallelDesc) {
  op_graph.ForEachNode([&](OpNode* op_node) {
    const VariableOp* var_op = dynamic_cast<const VariableOp*>(&op_node->op());
    if (var_op == nullptr) { return; }
    if (lbi2diff_lbi.find(var_op->BnInOp2Lbi(var_op->SoleObn())) == lbi2diff_lbi.end()) { return; }

    LogicalBlobId diff_lbi_of_var_out = lbi2diff_lbi.at(var_op->BnInOp2Lbi(var_op->SoleObn()));
    const auto& parallel_desc = op_node->parallel_desc();
    GenerateDownScaleOpConf(var_op->op_name() + "-down_scale", parallel_desc.parallel_conf(),
                            job_builder, &diff_lbi_of_var_out);
    GenerateOptimizerOpConfIf(*var_op, parallel_desc.parallel_conf(), job_builder,
                              diff_lbi_of_var_out, LossInstanceNum4ParallelDesc(parallel_desc));
  });
}

template<typename T>
void ConstructMdUpdtOpConf(const VariableOp& op, const LogicalBlobId& diff_lbi_of_var_out,
                           const LogicalBlobId& total_loss_instance_num_lbi,
                           JobBuilder* job_builder, T* mdupdt_op_conf) {
  const auto& train_conf = job_builder->job().job_conf().train_conf();
  *mdupdt_op_conf->mutable_user_conf() = op.op_conf().variable_conf().has_model_update_conf()
                                             ? op.op_conf().variable_conf().model_update_conf()
                                             : train_conf.model_update_conf();
  mdupdt_op_conf->set_model_diff(GenLogicalBlobName(diff_lbi_of_var_out));
  mdupdt_op_conf->set_total_instance_num_diff(GenLogicalBlobName(total_loss_instance_num_lbi));
  mdupdt_op_conf->set_model(GenLogicalBlobName(op.BnInOp2Lbi("out")));
  mdupdt_op_conf->set_train_step(train_conf.train_step_lbn());

  if (op.op_conf().variable_conf().has_normalize_conf()) {
    mdupdt_op_conf->mutable_normalize_conf()->set_epsilon(
        op.op_conf().variable_conf().normalize_conf().epsilon());
    mdupdt_op_conf->mutable_normalize_conf()->set_axis(
        op.op_conf().variable_conf().normalize_conf().axis());
  }

  if (op.op_conf().variable_conf().model_name() == "weight") {
    const std::string& primary_lr_lbn =
        op.op_conf().variable_conf().has_learning_rate()
            ? AddScheduleOp(job_builder, mdupdt_op_conf->user_conf(), train_conf.train_step_lbn(),
                            "Variable-Train-PrimaryLearningRate-Scheduler" + op.op_conf().name(),
                            op.op_conf().variable_conf().learning_rate())
            : train_conf.primary_lr_lbn();
    const float l1 = op.op_conf().variable_conf().has_l1() ? op.op_conf().variable_conf().l1()
                                                           : train_conf.weight_l1();
    const float l2 = op.op_conf().variable_conf().has_l2() ? op.op_conf().variable_conf().l2()
                                                           : train_conf.weight_l2();
    mdupdt_op_conf->set_learning_rate(primary_lr_lbn);
    mdupdt_op_conf->set_l1(l1);
    mdupdt_op_conf->set_l2(l2);
  } else if (op.op_conf().variable_conf().model_name() == "bias") {
    const std::string& secondary_lr_lbn =
        op.op_conf().variable_conf().has_learning_rate()
            ? AddScheduleOp(job_builder, mdupdt_op_conf->user_conf(), train_conf.train_step_lbn(),
                            "Variable-Train-SecondaryLearningRate-Scheduler" + op.op_conf().name(),
                            op.op_conf().variable_conf().learning_rate())
            : train_conf.secondary_lr_lbn();
    const float l1 = op.op_conf().variable_conf().has_l1() ? op.op_conf().variable_conf().l1()
                                                           : train_conf.bias_l1();
    const float l2 = op.op_conf().variable_conf().has_l2() ? op.op_conf().variable_conf().l2()
                                                           : train_conf.bias_l2();
    mdupdt_op_conf->set_learning_rate(secondary_lr_lbn);
    mdupdt_op_conf->set_l1(l1);
    mdupdt_op_conf->set_l2(l2);
  } else {
    const std::string& primary_lr_lbn =
        op.op_conf().variable_conf().has_learning_rate()
            ? AddScheduleOp(job_builder, mdupdt_op_conf->user_conf(), train_conf.train_step_lbn(),
                            "Variable-Train-PrimaryLearningRate-Scheduler" + op.op_conf().name(),
                            op.op_conf().variable_conf().learning_rate())
            : train_conf.primary_lr_lbn();
    const float l1 = op.op_conf().variable_conf().has_l1() ? op.op_conf().variable_conf().l1() : 0;
    const float l2 = op.op_conf().variable_conf().has_l2() ? op.op_conf().variable_conf().l2() : 0;
    mdupdt_op_conf->set_learning_rate(primary_lr_lbn);
    mdupdt_op_conf->set_l1(l1);
    mdupdt_op_conf->set_l2(l2);
  }
}

#define INSTANTIATE_CONSTRUCTOR_MDUPDT_OP_CONF(T)                                          \
  template void ConstructMdUpdtOpConf<T>(const VariableOp& op,                             \
                                         const LogicalBlobId& diff_lbi_of_var_out,         \
                                         const LogicalBlobId& total_loss_instance_num_lbi, \
                                         JobBuilder* job_builder, T* mdupdt_op_conf)

INSTANTIATE_CONSTRUCTOR_MDUPDT_OP_CONF(NaiveModelUpdateOpConf);
INSTANTIATE_CONSTRUCTOR_MDUPDT_OP_CONF(MomentumModelUpdateOpConf);
INSTANTIATE_CONSTRUCTOR_MDUPDT_OP_CONF(RMSPropModelUpdateOpConf);
INSTANTIATE_CONSTRUCTOR_MDUPDT_OP_CONF(LARSModelUpdateOpConf);
INSTANTIATE_CONSTRUCTOR_MDUPDT_OP_CONF(AdamModelUpdateOpConf);
INSTANTIATE_CONSTRUCTOR_MDUPDT_OP_CONF(LazyAdamModelUpdateOpConf);

}  // namespace oneflow
