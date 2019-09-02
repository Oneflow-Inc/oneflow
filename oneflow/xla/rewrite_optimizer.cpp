#include "glog/logging.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include <string>
#include <vector>

#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/job_builder.h"
#include "oneflow/xla/of2xla/xla_utility.h"
#include "oneflow/xla/of2xla/xla_graph.h"
#include "oneflow/xla/of2xla/xla_node_attr.h"
#include "oneflow/xla/rewrite_optimizer_util.h"
#include "oneflow/xla/rewrite_optimizer.h"

namespace oneflow {

// Rewrite model update ops to optimizer graphs
class OptimizerRewritor {
 public:
  OptimizerRewritor(const mola::XlaGraph &graph, Job *job)
      : graph_(graph), builder_(std::make_shared<JobBuilder>(job)) {}

  virtual void Run();

 private:
  OptimizerMode GetOptimizerModeIfModelUpdate(const mola::XlaNode *node) const;

  OperatorConf *BuildClipGradientOp(const std::string &node_name,
                                    const std::string &gradient,
                                    const std::string &total_instances,
                                    const ClipConf &clip_conf);

  OperatorConf *BuildLearningRateShedulerOp(
                                const std::string &node_name,
                                const float learning_rate,
                                const NormalModelUpdateOpUserConf &update_conf);

  OperatorConf *BuildOptimizerOp(const mola::XlaNode *node,
                                 const std::string &gradient,
                                 const std::string &total_instances,
                                 const std::string &learning_rate);

  OperatorConf *BuildFakeConsumeOp(const std::string &node_name,
                                   const std::string &input);

  std::vector<std::string> GetControlInOpNames(const mola::XlaNode *node) const;

  const mola::XlaGraph &graph_;
  std::shared_ptr<JobBuilder> builder_;
};

OptimizerMode OptimizerRewritor::GetOptimizerModeIfModelUpdate(
    const mola::XlaNode *node) const {
  if (node->op_type() == "NavieModelUpdate") {
    return OptimizerMode::kNaive;
  } else if (node->op_type() == "MomentumModelUpdate") {
    return OptimizerMode::kMomentum;
  } else if (node->op_type() == "RMSPropModelUpdate") {
    return OptimizerMode::kRMSProp;
  } else if (node->op_type() == "LARSModelUpdate") {
    return OptimizerMode::kLARS;
  } else if (node->op_type() == "AdamModelUpdate") {
    return OptimizerMode::kAdam;
  } else {
    return OptimizerMode::kInvalid;
  }
}

OperatorConf *OptimizerRewritor::BuildClipGradientOp(
                                const std::string &node_name,
                                const std::string &gradient,
                                const std::string &total_instances,
                                const ClipConf &clip_conf) {
  OperatorConf op_conf;
  op_conf.set_name(absl::StrCat(node_name, "-clip_gradient"));
  ClipGradientOpConf *conf = op_conf.mutable_clip_gradient_conf();
  conf->set_out("out");
  conf->set_gradient(gradient);
  conf->set_instance_num_diff(total_instances);
  conf->set_clip_norm(clip_conf.clip_by_global_norm().clip_norm());

  if (clip_conf.clip_by_global_norm().has_global_norm()) {
    float global_norm = clip_conf.clip_by_global_norm().global_norm();
    conf->set_global_norm(global_norm);
  }
  
  ParallelConf parallel_conf = builder_->GetParallelConf(node_name);
  builder_->AddOps(parallel_conf, {op_conf});
  return builder_->MutableOpConf(op_conf.name());
}

OperatorConf *OptimizerRewritor::BuildLearningRateShedulerOp(
                            const std::string &node_name,
                            const float learning_rate,
                            const NormalModelUpdateOpUserConf &update_conf) {
  OperatorConf op_conf;
  op_conf.set_name(absl::StrCat(node_name, "-lr_sheduler"));
  LearningRateShedulerOpConf *conf = op_conf.mutable_lr_sheduler_conf();
  conf->set_out("out");
  conf->set_base_learning_rate(learning_rate);

  if (update_conf.has_warmup_conf()) {
    conf->set_use_warmup(true);
    OptimizerParamBuilder::SetupWarmupParam(conf, update_conf.warmup_conf());
  } else {
    conf->set_use_warmup(false);
  }
  if (update_conf.has_learning_rate_decay()) {
    const auto &lr_decay_conf = update_conf.learning_rate_decay();
    OptimizerParamBuilder::SetupLearningRateDecayParam(conf, lr_decay_conf);
  }

  ParallelConf parallel_conf = builder_->GetParallelConf(node_name);
  builder_->AddOps(parallel_conf, {op_conf});
  return builder_->MutableOpConf(op_conf.name());
}

OperatorConf *OptimizerRewritor::BuildOptimizerOp(
                                const mola::XlaNode *node,
                                const std::string &gradient,
                                const std::string &total_instances,
                                const std::string &learning_rate) {
  OptimizerMode mode = GetOptimizerModeIfModelUpdate(node);
  CHECK_NE(mode, OptimizerMode::kInvalid);
  OperatorConf op_conf = OptimizerParamBuilder::Build(
      mode, node, gradient, total_instances, learning_rate);

  ParallelConf parallel_conf = builder_->GetParallelConf(node->op_name());
  builder_->AddOrMutOps(parallel_conf, {op_conf});
  return builder_->MutableOpConf(op_conf.name());
}

OperatorConf *OptimizerRewritor::BuildFakeConsumeOp(
                                const std::string &node_name,
                                const std::string &input) {
  OperatorConf op_conf;
  op_conf.set_name(absl::StrCat(node_name, "-fake_consume"));
  op_conf.mutable_fake_consume_conf()->add_in(input);

  ParallelConf parallel_conf = builder_->GetParallelConf(node_name);
  builder_->AddOps(parallel_conf, {op_conf});
  return builder_->MutableOpConf(op_conf.name());
}

std::vector<std::string> OptimizerRewritor::GetControlInOpNames(
    const mola::XlaNode *node) const {
  const auto &op_conf = builder_->GetOpConf(node->op_name());
  std::vector<std::string> ctrl_in_op_names;
  for (const std::string &name : op_conf.ctrl_in_op_name()) {
    ctrl_in_op_names.push_back(name);
  }
  return std::move(ctrl_in_op_names);
}

void SetControlInOpNames(OperatorConf *op_conf,
                         const std::vector<std::string> &ctrl_in_op_names) {
  for (const std::string &name : ctrl_in_op_names) {
    // op_conf->mutable_ctrl_in_op_name()->Add()->assign(name);
    op_conf->add_ctrl_in_op_name(name);
  }
}

void OptimizerRewritor::Run() {
  for (const mola::XlaNode *node : graph_.Nodes()) {
    OptimizerMode mode = GetOptimizerModeIfModelUpdate(node);
    if (mode == OptimizerMode::kInvalid) {
      // Skip the node if it is not a model update node
      continue;
    }
    using mola::GetNodeAttr;
    float learning_rate = GetNodeAttr<float>(node, "learning_rate");
    std::string model_diff = GetNodeAttr<std::string>(node, "model_diff");
    std::string total_instances =
        GetNodeAttr<std::string>(node, "total_instance_num_diff");

    auto control_in_op_names = GetControlInOpNames(node);
    std::vector<OperatorConf *> operator_confs;
    const auto *user_conf = dynamic_cast<NormalModelUpdateOpUserConf *>(
        GetNodeAttr<PbMessage *>(node, "user_conf"));
    CHECK_NOTNULL(user_conf);
    // Create clip gradient operator if `has_clip_conf`
    if (user_conf->has_clip_conf()) {
      OperatorConf *clip_conf = BuildClipGradientOp(node->op_name(), model_diff,
                                                    total_instances,
                                                    user_conf->clip_conf());
      operator_confs.push_back(clip_conf);
      model_diff = absl::StrCat(clip_conf->name(), "/out");
    }

    // TODO(hjchen2): learning_rate maybe a untrainable variable
    // Always build a learning rate sheduler operator even if using const
    // learning rate
    OperatorConf *lr_shedule_conf = BuildLearningRateShedulerOp(
        node->op_name(), learning_rate, *user_conf);
    operator_confs.push_back(lr_shedule_conf);

    std::string lr_shedule_output =
        absl::StrCat(lr_shedule_conf->name(), "/out");
    OperatorConf *optimizer_conf = BuildOptimizerOp(
        node, model_diff, total_instances, lr_shedule_output);
    operator_confs.push_back(optimizer_conf);

    // Currently each model update operator will result in a fake consumer
    // TODO(hjchen2): Only one global final fake consume operator maybe better.
    BuildFakeConsumeOp(node->op_name(),
                       absl::StrCat(optimizer_conf->name(), "/out"));

    if (control_in_op_names.size() > 0) {
      for (OperatorConf *op_conf : operator_confs) {
        // control_in_op_names.push_back(op_conf->name());
        SetControlInOpNames(op_conf, control_in_op_names);
      }
    }
  }
}

void RewriteOptimizerGraph(const mola::XlaGraph &graph, Job *job) {
  OptimizerRewritor(graph, job).Run();
}

}  // namespace oneflow
