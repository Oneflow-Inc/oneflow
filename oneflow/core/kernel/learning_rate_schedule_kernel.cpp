/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"

namespace oneflow {

class LearningRateScheduleKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LearningRateScheduleKernel);
  LearningRateScheduleKernel() = default;
  ~LearningRateScheduleKernel() override = default;

 private:
  void VirtualKernelInit(KernelContext* ctx) override {
    if (Global<ResourceDesc, ForSession>::Get()->enable_debug_mode()) {
      log_stream_ = TeePersistentLogStream::Create("train_step2lr.csv");
      (*log_stream_) << "train_step, lr\n";
    }
  }
  void ForwardDataContent(KernelContext* ctx) const override;

  std::unique_ptr<TeePersistentLogStream> log_stream_;
};

namespace {

double ConstantWarmupLearningRate(const ConstantWarmupConf& conf, double lr, int64_t train_step) {
  CHECK_GE(conf.warmup_batches(), 0);
  CHECK_GT(conf.multiplier(), 0);
  CHECK_LT(conf.multiplier(), 1);
  if (train_step < conf.warmup_batches()) {
    return lr * conf.multiplier();
  } else {
    return lr;
  }
}

double LinearWarmupLearningRate(const LinearWarmupConf& conf, double lr, int64_t train_step) {
  CHECK_GE(conf.warmup_batches(), 0);
  CHECK_GE(conf.start_multiplier(), 0);
  CHECK_LT(conf.start_multiplier(), 1);
  double start_multiplier = conf.start_multiplier();
  double multiplier = 1.0;
  if (train_step < conf.warmup_batches()) {
    multiplier =
        start_multiplier + (1.0 - start_multiplier) * (train_step * 1.0 / conf.warmup_batches());
  }
  return lr * multiplier;
}

bool TriggerWarmup(const LearningRateScheduleOpConf& conf, double lr, int64_t train_step) {
  if (!conf.has_warmup_conf()) { return false; }
  const WarmupConf& warmup_conf = conf.warmup_conf();
  if (warmup_conf.has_constant_conf()) {
    return (train_step < warmup_conf.constant_conf().warmup_batches());
  } else if (warmup_conf.has_linear_conf()) {
    return (train_step < warmup_conf.linear_conf().warmup_batches());
  } else {
    UNIMPLEMENTED();
  }
}

double GetWarmupLearningRate(const WarmupConf& conf, double lr, int64_t train_step) {
  if (conf.has_constant_conf()) {
    return ConstantWarmupLearningRate(conf.constant_conf(), lr, train_step);
  } else if (conf.has_linear_conf()) {
    return LinearWarmupLearningRate(conf.linear_conf(), lr, train_step);
  } else {
    UNIMPLEMENTED();
  }
}

double ExponentialDecayedLearningRate(const ExponentialDecayConf& conf, double lr,
                                      int64_t cur_batch_num) {
  CHECK_GT(conf.decay_batches(), 0);
  double p = static_cast<double>(cur_batch_num) / static_cast<double>(conf.decay_batches());
  if (conf.staircase()) { p = std::floor(p); }
  return lr * std::pow(conf.decay_rate(), p);
}

double InverseTimeDecayedLearningRate(const InverseTimeDecayConf& conf, double lr,
                                      int64_t cur_batch_num) {
  CHECK_GT(conf.decay_batches(), 0);
  double p = static_cast<double>(cur_batch_num) / static_cast<double>(conf.decay_batches());
  if (conf.staircase()) { p = std::floor(p); }
  return lr / (1.0 + conf.decay_rate() * p);
}

double NaturalExpDecayedLearningRate(const NaturalExpDecayConf& conf, double lr,
                                     int64_t cur_batch_num) {
  CHECK_GT(conf.decay_batches(), 0);
  double p = static_cast<double>(cur_batch_num) / static_cast<double>(conf.decay_batches());
  if (conf.staircase()) { p = std::floor(p); }
  return lr * std::exp(-conf.decay_rate() * p);
}

double PiecewiseConstantLearningRate(const PiecewiseConstantConf& conf, double lr,
                                     int64_t cur_batch_num) {
  const PbRf<int64_t>& boundaries = conf.boundaries();
  const PbRf<double>& values = conf.values();
  CHECK_EQ(boundaries.size() + 1, values.size());
  size_t i = 0;
  for (; i < boundaries.size(); ++i) {
    if (cur_batch_num <= boundaries[i]) { break; }
  }
  return values[i];
}

double PolynomialDecayedLearningRate(const PolynomialDecayConf& conf, double lr,
                                     int64_t cur_batch_num) {
  CHECK_GT(conf.decay_batches(), 0);
  double cur_batch = static_cast<double>(cur_batch_num);
  double decay_batches = static_cast<double>(conf.decay_batches());
  if (conf.cycle()) {
    if (cur_batch_num == 0) { cur_batch = 1.0; }
    decay_batches = decay_batches * std::ceil(cur_batch / decay_batches);
  } else {
    cur_batch = std::min(cur_batch, decay_batches);
  }
  return (lr - conf.end_learning_rate()) * std::pow(1.0 - (cur_batch / decay_batches), conf.power())
         + conf.end_learning_rate();
}

double CosineDecayedLearningRate(const CosineDecayConf& conf, double lr, int64_t cur_batch_num) {
  CHECK_GT(conf.decay_batches(), 0);
  const double PI = std::atan(1.0) * 4.0;
  double cur_batch = static_cast<double>(cur_batch_num);
  double decay_batches = static_cast<double>(conf.decay_batches());
  cur_batch = std::min(cur_batch, decay_batches);
  double cosine_decay = 0.5 * (1.0 + std::cos(PI * cur_batch / decay_batches));
  double decayed = (1.0 - conf.alpha()) * cosine_decay + conf.alpha();
  return lr * decayed;
}

double CosineAnnealingDecayedLearningRate(const CosineAnnealingDecayConf& conf, double lr,
                                          int64_t cur_batch_num) {
  CHECK_GT(conf.t_max(), 0);
  if (0 == cur_batch_num) { return lr; }

  const double PI = std::atan(1.0) * 4.0;
  const double eta_min = conf.eta_min();
  CHECK_LT(eta_min, lr);
  const double t_max_d = static_cast<double>(conf.t_max());
  const double cur_batch_num_d = static_cast<double>(cur_batch_num);

  return eta_min + (((lr - eta_min) * (1 + std::cos(PI * (cur_batch_num_d / t_max_d)))) / 2);
}

double LinearCosineDecayedLearningRate(const LinearCosineDecayConf& conf, double lr,
                                       int64_t cur_batch_num) {
  CHECK_GT(conf.decay_batches(), 0);
  const double PI = std::atan(1.0) * 4.0;
  double cur_batch = static_cast<double>(cur_batch_num);
  double decay_batches = static_cast<double>(conf.decay_batches());
  cur_batch = std::min(cur_batch, decay_batches);
  double linear_decay = (decay_batches - cur_batch) / decay_batches;
  double cosine_decay =
      0.5 * (1.0 + std::cos(PI * 2.0 * conf.num_periods() * cur_batch / decay_batches));
  double decayed = (conf.alpha() + linear_decay) * cosine_decay + conf.beta();
  return lr * decayed;
}

double PiecewiseScalingLearningRate(const PiecewiseScalingConf& conf, double lr,
                                    int64_t cur_batch_num) {
  const PbRf<int64_t>& boundaries = conf.boundaries();
  const PbRf<double>& scales = conf.scales();
  CHECK_EQ(boundaries.size() + 1, scales.size());
  size_t i = 0;
  for (; i < boundaries.size(); ++i) {
    if (cur_batch_num <= boundaries[i]) { break; }
  }
  return scales[i] * lr;
}

double StepLearningRate(const StepConf& conf, double lr, int64_t cur_batch_num) {
  const int64_t step_size = conf.step_size();
  CHECK_GE(step_size, 1);
  const double gamma = conf.gamma();

  double cur_batch = static_cast<double>(cur_batch_num);
  double step = static_cast<double>(step_size);
  size_t i = std::floor(cur_batch / step);

  return lr * std::pow(gamma, i);
}

double MultiStepLearningRate(const MultiStepConf& conf, double lr, int64_t cur_batch_num) {
  const PbRf<int64_t>& milestones = conf.milestones();
  CHECK_GE(milestones.size(), 1);
  const double gamma = conf.gamma();

  size_t i = 0;
  if (cur_batch_num < milestones[milestones.size() - 1]) {
    for (; i < milestones.size(); ++i) {
      if (cur_batch_num < milestones[i]) { break; }
    }
  } else {
    i = milestones.size();
  }

  return lr * std::pow(gamma, i);
}

double GetDecayedLearningRate(const LearningRateDecayConf& conf, double lr, int64_t cur_batch_num) {
  if (conf.has_exponential_conf()) {
    return ExponentialDecayedLearningRate(conf.exponential_conf(), lr, cur_batch_num);
  } else if (conf.has_inverse_time_conf()) {
    return InverseTimeDecayedLearningRate(conf.inverse_time_conf(), lr, cur_batch_num);
  } else if (conf.has_natural_exp_conf()) {
    return NaturalExpDecayedLearningRate(conf.natural_exp_conf(), lr, cur_batch_num);
  } else if (conf.has_piecewise_constant_conf()) {
    return PiecewiseConstantLearningRate(conf.piecewise_constant_conf(), lr, cur_batch_num);
  } else if (conf.has_polynomial_conf()) {
    return PolynomialDecayedLearningRate(conf.polynomial_conf(), lr, cur_batch_num);
  } else if (conf.has_cosine_conf()) {
    return CosineDecayedLearningRate(conf.cosine_conf(), lr, cur_batch_num);
  } else if (conf.has_cosine_annealing_conf()) {
    return CosineAnnealingDecayedLearningRate(conf.cosine_annealing_conf(), lr, cur_batch_num);
  } else if (conf.has_linear_cosine_conf()) {
    return LinearCosineDecayedLearningRate(conf.linear_cosine_conf(), lr, cur_batch_num);
  } else if (conf.has_piecewise_scaling_conf()) {
    return PiecewiseScalingLearningRate(conf.piecewise_scaling_conf(), lr, cur_batch_num);
  } else if (conf.has_step_conf()) {
    return StepLearningRate(conf.step_conf(), lr, cur_batch_num);
  } else if (conf.has_multi_step_conf()) {
    return MultiStepLearningRate(conf.multi_step_conf(), lr, cur_batch_num);
  } else {
    UNIMPLEMENTED();
  }
}

}  // namespace

void LearningRateScheduleKernel::ForwardDataContent(KernelContext* ctx) const {
  const LearningRateScheduleOpConf& conf = this->op_conf().learning_rate_schedule_conf();
  const int64_t train_step = *ctx->BnInOp2Blob("train_step")->dptr<int64_t>();
  float learning_rate = conf.learning_rate();
  if (TriggerWarmup(conf, learning_rate, train_step)) {
    learning_rate = GetWarmupLearningRate(conf.warmup_conf(), learning_rate, train_step);
  } else if (conf.has_learning_rate_decay()) {
    learning_rate = GetDecayedLearningRate(conf.learning_rate_decay(), learning_rate, train_step);
  }
  *ctx->BnInOp2Blob("out")->mut_dptr<float>() = learning_rate;

  if (Global<ResourceDesc, ForSession>::Get()->enable_debug_mode()) {
    (*log_stream_) << std::to_string(train_step) << ", " << std::to_string(learning_rate) << "\n";
    log_stream_->Flush();
  }
}

REGISTER_KERNEL(OperatorConf::kLearningRateScheduleConf, LearningRateScheduleKernel);

}  // namespace oneflow
