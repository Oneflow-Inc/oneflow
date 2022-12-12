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
#include <sys/types.h>
#include <unistd.h>

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/graph_scope_vars.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"

namespace oneflow {

class LearningRateScheduleKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LearningRateScheduleKernel);
  LearningRateScheduleKernel() = default;
  ~LearningRateScheduleKernel() override = default;

 private:
  void VirtualKernelInit(KernelContext* ctx) override {
    if (Singleton<ResourceDesc, ForSession>::Get()->enable_debug_mode()) {
      pid_t pid = getpid();
      log_stream_ = TeePersistentLogStream::Create(std::to_string(pid) + "-train_step2lr.csv");
      (*log_stream_) << "train_step, lr\n";
    }
    if (IsOpenGraphVerboseStepLr()) { print_step_lr_ = true; }
  }

  void ForwardDataContent(KernelContext* ctx) const override;
  bool print_step_lr_ = false;
  std::unique_ptr<TeePersistentLogStream> log_stream_;
};

namespace {

double GetDecayedLearningRate(const LearningRateDecayConf& conf, double base_lr, int64_t step);

double ConstantLearningRate(double base_lr, double factor, int64_t total_step, int64_t cur_step) {
  CHECK_GE(total_step, 0);
  CHECK_GE(factor, 0.0);
  CHECK_LE(factor, 1.0);
  if (cur_step < total_step) { return base_lr * factor; }
  return base_lr;
}

double LinearLearningRate(double base_lr, double start_factor, double end_factor,
                          int64_t total_step, int64_t cur_step) {
  CHECK_GE(total_step, 0);
  CHECK_GE(start_factor, 0.0);
  CHECK_LE(start_factor, 1.0);
  CHECK_GE(end_factor, 0.0);
  CHECK_LE(end_factor, 1.0);
  double multiplier = end_factor;
  double c_step_f = float(cur_step);
  double t_step_f = float(total_step);
  if (cur_step < total_step) {
    multiplier = start_factor + (end_factor - start_factor) * (c_step_f / t_step_f);
  }
  return base_lr * multiplier;
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

double CyclicLearningRate(const CyclicLRConf& conf, int64_t cur_batch_num) {
  double lr = 0;
  double gamma = conf.gamma();
  double base_lr = conf.base_lr();
  double max_lr = conf.base_lr();
  int64_t step_size_total = conf.step_size_total();
  double step_up_ratio =
      static_cast<double>(conf.step_size_up()) / static_cast<double>(step_size_total);
  double offset_ratio =
      static_cast<double>(cur_batch_num % step_size_total) / static_cast<double>(step_size_total);
  double scale_factor = 0;
  if (offset_ratio < step_up_ratio) {
    scale_factor = offset_ratio / step_up_ratio;
  } else {
    scale_factor = (1 - offset_ratio) / (1 - step_up_ratio);
  }

  int32_t mode = conf.mode();
  std::function<double(int)> scale_fn;
  if (mode == 0) {  // "triangular"
    scale_fn = [](int x) { return 1.0; };
  } else if (mode == 1) {  // "triangular2"
    scale_fn = [](int x) { return 1.0 / std::pow(2.0, (x - 1)); };
  } else if (mode == 2) {  // "exp_range"
    scale_fn = [&gamma](int x) { return std::pow(gamma, static_cast<double>(x)); };
  }

  int32_t scale_mode = conf.scale_mode();
  double height = (max_lr - base_lr) * scale_factor;
  if (scale_mode == 0) {  // "cycle"
    int32_t cycle_num = 1.0 + cur_batch_num / step_size_total;
    lr = base_lr + height * scale_fn(cycle_num);
  } else {
    lr = base_lr + height * scale_fn(cur_batch_num);
  }
  return lr;
}

double OneCycleLR(const OneCycleLRConf& conf, int64_t step) {
  std::function<double(double, double, double)> anneal_fn;
  if (conf.anneal_strategy() == 0) {  // "cos"
    anneal_fn = [](double start, double end, double pct) {
      double offset = (std::cos(M_PI * pct) + static_cast<double>(1)) / 2.0;
      return end + (end - start) * offset;
    };
  } else if (conf.anneal_strategy() == 1) {  // "linear"
    anneal_fn = [](double start, double end, double pct) { return start + (end - start) * pct; };
  }
  double max_lr = conf.max_lr();
  double initial_lr = max_lr / conf.div_factor();
  double min_lr = initial_lr / conf.final_div_factor();
  double total_steps = static_cast<double>(conf.total_steps());
  double milestone = conf.pct_start() * total_steps - 1;
  double step_double = static_cast<double>(step);
  double pct = 0;
  double lr = 0;
  if (conf.three_phase()) {
    if (lr < milestone) {
      pct = step_double / milestone;
      lr = anneal_fn(initial_lr, max_lr, pct);
    } else if (lr < milestone * 2) {
      pct = step_double / milestone - 1;
      lr = anneal_fn(max_lr, initial_lr, pct);
    } else {
      pct = (step_double - milestone * 2) / (total_steps - 1);
      lr = anneal_fn(initial_lr, min_lr, pct);
    }
  } else {
    if (lr < milestone) {
      pct = step_double / milestone;
      lr = anneal_fn(initial_lr, max_lr, pct);
    } else {
      pct = (step_double - milestone) / (total_steps - 1);
      lr = anneal_fn(max_lr, min_lr, pct);
    }
  }
  return lr;
}

double CosineAnnealingWarmRestartsLearningRate(const CosineAnnealingWarmRestartsConf& conf,
                                               const double base_lr, const int64_t step) {
  int64_t epoch_steps = conf.t_initial();
  int64_t epoch = step / epoch_steps;
  int64_t step_in_epoch = step - (epoch_steps * epoch);
  if (conf.t_mult() > 1) {
    epoch = static_cast<int64_t>(std::floor(
        std::log(1 - step / conf.t_initial() * (1 - conf.t_mult())) / std::log(conf.t_mult())));
    int64_t interval = std::pow(conf.t_mult(), epoch);
    epoch_steps = interval * conf.t_initial();
    step_in_epoch = step
                    - static_cast<int64_t>(std::floor(static_cast<double>(1 - interval)
                                                      / (1 - conf.t_mult()) * conf.t_initial()));
  }
  double lr = conf.eta_min();
  if (conf.restart_limit() == 0 || (conf.restart_limit() > 0 && epoch < conf.restart_limit())) {
    double gamma = std::pow(conf.decay_rate(), epoch);
    lr = lr + 0.5 * (base_lr * gamma - lr) * (1 + std::cos(M_PI * step_in_epoch / epoch_steps));
  }
  return lr;
}

double SequentialScheduler(const SequentialSchedulerConf& conf, const double base_lr,
                           const int64_t step) {
  CHECK_GE(conf.schedulers_size(), 1);
  CHECK_EQ(conf.milestones_size(), conf.schedulers_size() - 1);
  CHECK_EQ(conf.interval_rescaling_size(), conf.milestones_size());

  int64_t cur_step = step;
  size_t scheduler_idx = 0;
  for (size_t i = 0; i < conf.milestones_size(); ++i) {
    if (step < conf.milestones(i)) {
      break;
    } else {
      if (conf.interval_rescaling(i)) { cur_step = step - conf.milestones(i); }
      scheduler_idx++;
    }
  }
  return GetDecayedLearningRate(conf.schedulers(scheduler_idx), base_lr, cur_step);
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
  } else if (conf.has_constant_lr_conf()) {
    return ConstantLearningRate(lr, conf.constant_lr_conf().factor(),
                                conf.constant_lr_conf().total_iters(), cur_batch_num);
  } else if (conf.has_linear_lr_conf()) {
    return LinearLearningRate(lr, conf.linear_lr_conf().start_factor(),
                              conf.linear_lr_conf().end_factor(),
                              conf.linear_lr_conf().total_iters(), cur_batch_num);
  } else if (conf.has_cosine_annealing_warm_restarts_conf()) {
    return CosineAnnealingWarmRestartsLearningRate(conf.cosine_annealing_warm_restarts_conf(), lr,
                                                   cur_batch_num);
  } else if (conf.has_sequential_scheduler_conf()) {
    return SequentialScheduler(conf.sequential_scheduler_conf(), lr, cur_batch_num);
  } else if (conf.has_cyclic_lr_conf()) {
    return CyclicLearningRate(conf.cyclic_lr_conf(), cur_batch_num);
  } else if (conf.has_onecycle_lr_conf()) {
    return OneCycleLR(conf.onecycle_lr_conf(), cur_batch_num);
  } else {
    UNIMPLEMENTED();
  }
}

}  // namespace

void LearningRateScheduleKernel::ForwardDataContent(KernelContext* ctx) const {
  const LearningRateScheduleOpConf& conf = this->op_conf().learning_rate_schedule_conf();
  const int64_t train_step = *ctx->BnInOp2Blob("train_step")->dptr<int64_t>();
  float learning_rate = conf.learning_rate();
  if (conf.has_learning_rate_decay()) {
    learning_rate = GetDecayedLearningRate(conf.learning_rate_decay(), learning_rate, train_step);
  }
  // NOTE(lixiang): Set verbose=True will print step and lr.
  if (unlikely(print_step_lr_)) {
    std::cout << "Last step " << train_step << " adjusting learning rate to " << learning_rate
              << std::endl;
  }
  *ctx->BnInOp2Blob("out")->mut_dptr<float>() = learning_rate;
  if (Singleton<ResourceDesc, ForSession>::Get()->enable_debug_mode()) {
    (*log_stream_) << std::to_string(train_step) << ", " << std::to_string(learning_rate) << "\n";
    log_stream_->Flush();
  }
}

REGISTER_KERNEL(OperatorConf::kLearningRateScheduleConf, LearningRateScheduleKernel);

}  // namespace oneflow
