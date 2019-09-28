#include "oneflow/engine/xla/of2xla/xla_node_attr.h"
#include "oneflow/engine/xla/rewrite_optimizer_util.h"

namespace oneflow {

template <OptimizerMode mode>
void OptimizerParamBuilder::BuilderImpl::ApplyBuild() {
  LOG(FATAL) << "Unimplement ApplyBuild.";
}

template <>
void OptimizerParamBuilder::BuilderImpl::ApplyBuild<OptimizerMode::kAdam>() {
  using mla::GetNodeAttr;
  AdamOptimizerOpConf *conf = op_conf_->mutable_adam_optimizer_conf();
  float l1_weight_decay = GetNodeAttr<float>(node_, "l1");
  float l2_weight_decay = GetNodeAttr<float>(node_, "l2");
  conf->set_l1(l1_weight_decay);
  conf->set_l2(l2_weight_decay);

  conf->set_out("out");
  conf->set_out_m("out_m");
  conf->set_out_v("out_v");
  conf->set_gradient(gradient_);
  conf->set_instance_num_diff(total_instances_);
  conf->set_learning_rate(learning_rate_);
  conf->set_weight(GetNodeAttr<std::string>(node_, "model"));
  conf->set_m(GetNodeAttr<std::string>(node_, "m"));
  conf->set_v(GetNodeAttr<std::string>(node_, "v"));

  // conf->set_beta1(GetNodeAttr<std::string>(node_, "beta1_t"));
  // conf->set_beta2(GetNodeAttr<std::string>(node_, "beta2_t"));
  const auto *user_conf = dynamic_cast<NormalModelUpdateOpUserConf *>(
        GetNodeAttr<PbMessage *>(node_, "user_conf"));
  CHECK_NOTNULL(user_conf);
  if (user_conf->has_adam_conf()) {
    const float epsilon = user_conf->adam_conf().epsilon();
    conf->set_epsilon(epsilon);
    const float beta1 = user_conf->adam_conf().beta1();
    conf->set_beta1(beta1);
    const float beta2 = user_conf->adam_conf().beta2();
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
    const OptimizerMode &mode,
    const mla::XlaNode *node,
    const std::string &gradient,
    const std::string &total_instances,
    const std::string &learning_rate) {
  OperatorConf op_conf;
  op_conf.set_name(node->op_name());
  ApplyOptimizerModeVisitor(mode, BuilderImpl(node, gradient, total_instances,
                                              learning_rate, &op_conf));
  return op_conf;
}

}  // namespace oneflow
