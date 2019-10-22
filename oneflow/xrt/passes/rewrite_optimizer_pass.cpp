#include <string>
#include <vector>
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "glog/logging.h"

#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/job_builder.h"
#include "oneflow/xrt/graph/graph.h"
#include "oneflow/xrt/graph/node_attr.h"
#include "oneflow/xrt/passes/pass.h"
#include "oneflow/xrt/passes/rewrite_optimizer.h"

namespace oneflow {
namespace xrt {

// Rewrite model update ops to optimizer graphs
class OptimizerRewritor {
 public:
  OptimizerRewritor(const xrt::XrtGraph &graph, Job *job)
      : graph_(graph), builder_(std::make_shared<JobBuilder>(job)) {}

  virtual void Run();

 private:
  OptimizerMode GetOptimizerModeIfModelUpdate(const xrt::XrtNode *node) const;

  OperatorConf *BuildClipGradientOp(const std::string &node_name,
                                    const std::string &gradient,
                                    const std::string &total_instances,
                                    const ClipConf &clip_conf);

  OperatorConf *BuildOptimizerOp(const xrt::XrtNode *node,
                                 const std::string &gradient,
                                 const std::string &total_instances,
                                 const std::string &learning_rate);

  std::vector<std::string> GetControlInOpNames(const xrt::XrtNode *node) const;

  const xrt::XrtGraph &graph_;
  std::shared_ptr<JobBuilder> builder_;
};

OptimizerMode OptimizerRewritor::GetOptimizerModeIfModelUpdate(
    const xrt::XrtNode *node) const {
  if (node->type() == "NavieModelUpdate") {
    return OptimizerMode::kNaive;
  } else if (node->type() == "MomentumModelUpdate") {
    return OptimizerMode::kMomentum;
  } else if (node->type() == "RMSPropModelUpdate") {
    return OptimizerMode::kRMSProp;
  } else if (node->type() == "LARSModelUpdate") {
    return OptimizerMode::kLARS;
  } else if (node->type() == "AdamModelUpdate") {
    return OptimizerMode::kAdam;
  } else {
    return OptimizerMode::kInvalid;
  }
}

OperatorConf *OptimizerRewritor::BuildClipGradientOp(
    const std::string &node_name, const std::string &gradient,
    const std::string &total_instances, const ClipConf &clip_conf) {
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

  const ParallelConf &parallel_conf = builder_->ParallelConf4OpName(node_name);
  builder_->AddOps(parallel_conf, {op_conf});
  return builder_->MutableOpConf4OpName(op_conf.name());
}

OperatorConf *OptimizerRewritor::BuildOptimizerOp(
    const xrt::XrtNode *node, const std::string &gradient,
    const std::string &total_instances, const std::string &learning_rate) {
  OptimizerMode mode = GetOptimizerModeIfModelUpdate(node);
  CHECK(mode != OptimizerMode::kInvalid);
  OperatorConf op_conf = OptimizerParamBuilder::Build(
      mode, node, gradient, total_instances, learning_rate);

  const ParallelConf &parallel_conf =
      builder_->ParallelConf4OpName(node->name());
  builder_->AddOrMutOpsOnlyOnce(parallel_conf, {op_conf});
  return builder_->MutableOpConf4OpName(op_conf.name());
}

std::vector<std::string> OptimizerRewritor::GetControlInOpNames(
    const xrt::XrtNode *node) const {
  const auto &op_conf = builder_->OpConf4OpName(node->name());
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
  for (const xrt::XrtNode *node : graph_.Nodes()) {
    OptimizerMode mode = GetOptimizerModeIfModelUpdate(node);
    if (mode == OptimizerMode::kInvalid) {
      // Skip the node if it is not a model update node
      continue;
    }
    using xrt::GetNodeAttr;
    using xrt::GetNodeAttrAsString;
    std::string learning_rate = GetNodeAttrAsString(node, "learning_rate");
    std::string model_diff = GetNodeAttrAsString(node, "model_diff");
    std::string total_instances =
        GetNodeAttrAsString(node, "total_instance_num_diff");
    std::string train_step = GetNodeAttrAsString(node, "train_step");

    auto control_in_op_names = GetControlInOpNames(node);
    std::vector<OperatorConf *> operator_confs;
    const auto *user_conf = dynamic_cast<NormalModelUpdateOpUserConf *>(
        GetNodeAttr<PbMessage *>(node, "user_conf"));
    CHECK_NOTNULL(user_conf);
    // Create clip gradient operator if `has_clip_conf`
    if (user_conf->has_clip_conf()) {
      OperatorConf *clip_conf = BuildClipGradientOp(
          node->name(), model_diff, total_instances, user_conf->clip_conf());
      operator_confs.push_back(clip_conf);
      model_diff = absl::StrCat(clip_conf->name(), "/out");
    }

    OperatorConf *optimizer_conf =
        BuildOptimizerOp(node, model_diff, total_instances, learning_rate);
    operator_confs.push_back(optimizer_conf);

    if (control_in_op_names.size() > 0) {
      for (OperatorConf *op_conf : operator_confs) {
        // control_in_op_names.push_back(op_conf->name());
        SetControlInOpNames(op_conf, control_in_op_names);
      }
    }
  }
}

// Rewrite model update operator to optimizer graph
class RewriteOptimizerPass : public XrtPass {
 public:
  RewriteOptimizerPass() = default;

  // params: vector of any which should contains:
  //   0 - job
  void Run(XrtGraph *graph, const XrtPassOptions &options,
           const std::vector<Any> &params) override {
    CHECK_GE(params.size(), 1)
        << "Job is required by `RebuildCompiledJobPass`.";
    auto *job = any_cast<Job *>(params[0]);

    CHECK(graph) << "Graph is required by `RebuildCompiledJobPass`.";
    OptimizerRewritor(*graph, job).Run();
  }
};

REGISTER_XRT_PASS(RewriteOptimizer, RewriteOptimizerPass);

}  // namespace xrt
}  // namespace oneflow
