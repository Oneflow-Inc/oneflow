#include "oneflow/xrt/passes/rewrite_optimizer.h"
#include "oneflow/xrt/utility/message_attr.h"

namespace oneflow {
namespace xrt {

template <OptimizerMode mode>
void OptimizerParamBuilder::BuilderImpl::ApplyBuild() {
  LOG(FATAL) << "Unimplement ApplyBuild.";
}

template <>
void OptimizerParamBuilder::BuilderImpl::ApplyBuild<OptimizerMode::kAdam>() {
  AdamOptimizerOpConf *conf = op_conf_->mutable_adam_optimizer_conf();
  const AdamModelUpdateOpConf &update_conf =
      dynamic_cast<const OperatorConf *>(&node_->param())
          ->adam_model_update_conf();
  conf->set_m(update_conf.m());
  conf->set_v(update_conf.v());
  conf->set_weight(update_conf.model());

  conf->set_l1(update_conf.l1());
  conf->set_l2(update_conf.l2());

  conf->set_gradient(gradient_);
  conf->set_instance_num_diff(total_instances_);
  conf->set_learning_rate(learning_rate_);

  const NormalModelUpdateOpUserConf &user_conf = update_conf.user_conf();
  if (user_conf.has_adam_conf()) {
    const float epsilon = user_conf.adam_conf().epsilon();
    conf->set_epsilon(epsilon);
    const float beta1 = user_conf.adam_conf().beta1();
    conf->set_beta1(beta1);
    const float beta2 = user_conf.adam_conf().beta2();
    conf->set_beta2(beta2);
  }
}

/*static*/ void OptimizerParamBuilder::ApplyOptimizerModeVisitor(
    const OptimizerMode &mode, BuilderImpl builder) {
  switch (mode) {
    case OptimizerMode::kAdam:
      builder.ApplyBuild<OptimizerMode::kAdam>();
      break;
    default:
      LOG(FATAL) << "Unsupport OptimizerMode: " << mode;
  }
}

/*static*/ OperatorConf OptimizerParamBuilder::Build(
    const OptimizerMode &mode, const xrt::XrtNode *node,
    const std::string &gradient, const std::string &total_instances,
    const std::string &learning_rate) {
  OperatorConf op_conf;
  op_conf.set_name(node->name());
  ApplyOptimizerModeVisitor(mode, BuilderImpl(node, gradient, total_instances,
                                              learning_rate, &op_conf));
  return op_conf;
}

}  // namespace xrt
}  // namespace oneflow
