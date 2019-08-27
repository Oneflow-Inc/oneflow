#include "oneflow/xla/of2xla/xla_node_attr.h"
#include "oneflow/xla/rewrite_optimizer_util.h"

namespace oneflow {

template<typename ParamType>
void SetupLearningRateShedulerParam(LearningRateShedulerOpConf *conf,
                                    const ParamType &param) {
  conf->set_steps(param.decay_batches());
  conf->set_decay(param.decay_rate());
  conf->set_staircase(param.staircase()); 
}

template<>
void SetupLearningRateShedulerParam(LearningRateShedulerOpConf *conf,
                                    const PiecewiseConstantConf &param) {
  *(conf->mutable_boundaries()) = param.boundaries();
  *(conf->mutable_values()) = param.values();
}

template<>
void SetupLearningRateShedulerParam(LearningRateShedulerOpConf *conf,
                                    const PolynomialDecayConf &param) {
  conf->set_steps(param.decay_batches());
  conf->set_end_learning_rate(param.end_learning_rate());
  conf->set_power(param.power());
  conf->set_cycle(param.cycle());
}

template<>
void SetupLearningRateShedulerParam(LearningRateShedulerOpConf *conf,
                                    const CosineDecayConf &param) {
  conf->set_steps(param.decay_batches());
  conf->set_alpha(param.alpha());
}

template<>
void SetupLearningRateShedulerParam(LearningRateShedulerOpConf *conf,
                                    const LinearCosineDecayConf &param) {
  conf->set_steps(param.decay_batches());
  conf->set_alpha(param.alpha());
  conf->set_beta(param.beta());
  conf->set_num_periods(param.num_periods());
}

/*static*/ void OptimizerParamBuilder::SetupLearningRateDecayParam(
                            LearningRateShedulerOpConf *conf,
                            const LearningRateDecayConf &lr_decay_conf) {
  if (lr_decay_conf.has_exponential_conf()) {
    conf->set_decay_policy("exponential");
    SetupLearningRateShedulerParam<ExponentialDecayConf>(
        conf, lr_decay_conf.exponential_conf());
  } else if (lr_decay_conf.has_inverse_time_conf()) {
    conf->set_decay_policy("inverse");
    SetupLearningRateShedulerParam<InverseTimeDecayConf>(
        conf, lr_decay_conf.inverse_time_conf());
  } else if (lr_decay_conf.has_natural_exp_conf()) {
    conf->set_decay_policy("natural_exp");
    SetupLearningRateShedulerParam<NaturalExpDecayConf>(
        conf, lr_decay_conf.natural_exp_conf());
  } else if (lr_decay_conf.has_piecewise_constant_conf()) {
    conf->set_decay_policy("piecewise_constant");
    SetupLearningRateShedulerParam<PiecewiseConstantConf>(
        conf, lr_decay_conf.piecewise_constant_conf());
  } else if (lr_decay_conf.has_polynomial_conf()) {
    conf->set_decay_policy("polynomial");
    SetupLearningRateShedulerParam<PolynomialDecayConf>(
        conf, lr_decay_conf.polynomial_conf());
  } else if (lr_decay_conf.has_cosine_conf()) {
    conf->set_decay_policy("cosine");
    SetupLearningRateShedulerParam<CosineDecayConf>(
        conf, lr_decay_conf.cosine_conf());
  } else if (lr_decay_conf.has_linear_cosine_conf()) {
    conf->set_decay_policy("linear_cosine");
    SetupLearningRateShedulerParam<LinearCosineDecayConf>(
        conf, lr_decay_conf.linear_cosine_conf());
  }
}

/*static*/ void OptimizerParamBuilder::SetupWarmupParam(
                            LearningRateShedulerOpConf *conf,
                            const WarmupConf &warmup_conf) {
  int64_t warmup_steps = 0;
  if (warmup_conf.has_constant_conf()) {
    warmup_steps = warmup_conf.constant_conf().warmup_batches();
    double multiplier = warmup_conf.constant_conf().multiplier();
    conf->set_warmup_policy("Constant");
    conf->set_multiplier(multiplier);
  } else if (warmup_conf.has_linear_conf()) {
    warmup_steps = warmup_conf.linear_conf().warmup_batches();
    double start_multiplier = warmup_conf.linear_conf().start_multiplier();
    conf->set_warmup_policy("Linear");
    conf->set_warmup_steps(warmup_steps);
    conf->set_start_multiplier(start_multiplier);
  }
  conf->set_warmup_steps(warmup_steps);
}

template <OptimizerMode mode>
void OptimizerParamBuilder::BuilderImpl::ApplyBuild() {
  LOG(FATAL) << "Unimplement ApplyBuild.";
}

template <>
void OptimizerParamBuilder::BuilderImpl::ApplyBuild<OptimizerMode::kAdam>() {
  using mola::GetNodeAttr;
  AdamOptimizerOpConf *conf = op_conf_->mutable_adam_optimizer_conf();
  float l1_weight_decay = GetNodeAttr<float>(node_, "l1");
  float l2_weight_decay = GetNodeAttr<float>(node_, "l2");
  conf->set_l1(l1_weight_decay);
  conf->set_l2(l2_weight_decay);

  conf->set_gradient(gradient_);
  conf->set_instance_num_diff(total_instances_);
  conf->set_learning_rate(learning_rate_);
  conf->set_weight(GetNodeAttr<std::string>(node_, "model"));
  conf->set_m(GetNodeAttr<std::string>(node_, "m"));
  conf->set_v(GetNodeAttr<std::string>(node_, "v"));
  // conf->set_beta1(GetNodeAttr<std::string>(node_, "beta1_t"));
  // conf->set_beta2(GetNodeAttr<std::string>(node_, "beta2_t"));
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
                                        const mola::XlaNode *node,
                                        const std::string &gradient,
                                        const std::string &total_instances,
                                        const std::string &learning_rate) {
  OperatorConf op_conf;
  ApplyOptimizerModeVisitor(mode, BuilderImpl(node, gradient, total_instances,
                                              learning_rate, &op_conf));
  op_conf.set_name(node->op_name());
  return op_conf;
}

}  // namespace oneflow
