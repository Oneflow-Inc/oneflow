#include "oneflow/core/kernel/normal_model_update_kernel.h"
#include "oneflow/core/kernel/naive_model_update_kernel.h"
#include "oneflow/core/kernel/momentum_model_update_kernel.h"
#include "oneflow/core/kernel/rmsprop_model_update_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void NormalMdUpdateKernel<device_type, T>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto tpl = reinterpret_cast<std::tuple<int64_t, const Blob*>*>(ctx.other);
  int64_t next_model_vid = std::get<0>(*tpl);
  const NormalModelUpdateOpUserConf& conf =
      this->op_conf().normal_mdupdt_conf().user_conf();
  double learning_rate = conf.learning_rate();
  if (conf.has_learning_rate_decay()) {
    learning_rate = GetDecayedLearningRate(conf.learning_rate_decay(),
                                           learning_rate, next_model_vid - 1);
  }
  UpdateModel(ctx.device_ctx, std::get<1>(*tpl),
              DiffAveragingAndL1Regularization(ctx.device_ctx, BnInOp2Blob),
              next_model_vid, learning_rate, BnInOp2Blob);
}

template<DeviceType device_type, typename T>
Blob* NormalMdUpdateKernel<device_type, T>::DiffAveragingAndL1Regularization(
    DeviceCtx* ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_0 = BnInOp2Blob(this->kernel_conf().input_bns(0));
  FOR_RANGE(size_t, i, 1, this->kernel_conf().input_bns().size()) {
    Blob* in_i = BnInOp2Blob(this->kernel_conf().input_bns(i));
    KernelUtil<device_type, T>::Axpy(ctx, in_0->shape().elem_cnt(), 1.0,
                                     in_i->dptr<T>(), 1, in_0->mut_dptr<T>(),
                                     1);
  }
  const Blob* model = BnInOp2Blob("model");
  float l1 = Global<JobDesc>::Get()->L1();
  NormalMdUpdateKernelUtil<device_type, T>::DiffAveragingAndL1Regularization(
      ctx, model->shape().elem_cnt(), static_cast<T>(l1), model->dptr<T>(),
      in_0->mut_dptr<T>());
  return in_0;
}

template<typename T>
class NormalMdUpdateKernelUtil<DeviceType::kCPU, T> final {
 public:
  static void DiffAveragingAndL1Regularization(DeviceCtx* ctx, int64_t n, T l1,
                                               const T* model,
                                               T* model_diff_acc) {
    T zero = ZeroVal<T>::value;
    for (int64_t i = 0; i != n; ++i) {
      model_diff_acc[i] /= Global<JobDesc>::Get()->BatchSize();
      model_diff_acc[i] += l1 * ((model[i] >= zero) - (model[i] <= zero));
    }
  }
};

#define INSTANTIATE_KERNEL(device_type, data_type_pair) \
  template struct NormalMdUpdateKernel<device_type,     \
                                       OF_PP_PAIR_FIRST(data_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_KERNEL, DEVICE_TYPE_SEQ,
                                 FLOATING_DATA_TYPE_SEQ)

namespace {

Kernel* CreateMdUpdtKernel(const KernelConf& kernel_conf) {
  const NormalModelUpdateOpUserConf& user_conf =
      kernel_conf.op_conf().normal_mdupdt_conf().user_conf();
  if (user_conf.has_naive_conf()) {
    return CreateNaiveMdUpdtKernel(kernel_conf);
  } else if (user_conf.has_momentum_conf()) {
    return CreateMomentumMdUpdtKernel(kernel_conf);
  } else if (user_conf.has_rmsprop_conf()) {
    return CreateRMSPropMdUpdtKernel(kernel_conf);
  } else {
    UNIMPLEMENTED();
  }
}

double ExponentialDecayedLearningRate(const ExponentialDecayConf& conf,
                                      double lr, int64_t now_batch_num) {
  CHECK_GT(conf.decay_batches(), 0);
  double p = static_cast<double>(now_batch_num)
             / static_cast<double>(conf.decay_batches());
  if (conf.staircase()) { p = std::floor(p); }
  return lr * std::pow(conf.decay_rate(), p);
}

double InverseTimeDecayedLearningRate(const InverseTimeDecayConf& conf,
                                      double lr, int64_t now_batch_num) {
  CHECK_GT(conf.decay_batches(), 0);
  double p = static_cast<double>(now_batch_num)
             / static_cast<double>(conf.decay_batches());
  if (conf.staircase()) { p = std::floor(p); }
  return lr / (1.0 + conf.decay_rate() * p);
}

double NaturalExpDecayedLearningRate(const NaturalExpDecayConf& conf, double lr,
                                     int64_t now_batch_num) {
  CHECK_GT(conf.decay_batches(), 0);
  double p = static_cast<double>(now_batch_num)
             / static_cast<double>(conf.decay_batches());
  if (conf.staircase()) { p = std::floor(p); }
  return lr * std::exp(-conf.decay_rate() * p);
}

double PiecewiseConstantLearningRate(const PiecewiseConstantConf& conf,
                                     double lr, int64_t now_batch_num) {
  const PbRf<int64_t>& boundaries = conf.boundaries();
  const PbRf<double>& values = conf.values();
  CHECK_EQ(boundaries.size() + 1, values.size());
  size_t i = 0;
  for (; i < boundaries.size(); ++i) {
    if (now_batch_num <= boundaries[i]) { break; }
  }
  return values[i];
}

double PolynomialDecayedLearningRate(const PolynomialDecayConf& conf, double lr,
                                     int64_t now_batch_num) {
  CHECK_GT(conf.decay_batches(), 0);
  double now_batch = static_cast<double>(now_batch_num);
  double decay_batches = static_cast<double>(conf.decay_batches());
  if (conf.cycle()) {
    if (now_batch_num == 0) { now_batch = 1.0; }
    decay_batches = decay_batches * std::ceil(now_batch / decay_batches);
  } else {
    now_batch = std::min(now_batch, decay_batches);
  }
  return (lr - conf.end_learning_rate())
             * std::pow(1.0 - (now_batch / decay_batches), conf.power())
         + conf.end_learning_rate();
}

double CosineDecayedLearningRate(const CosineDecayConf& conf, double lr,
                                 int64_t now_batch_num) {
  CHECK_GT(conf.decay_batches(), 0);
  const double PI = std::atan(1.0) * 4.0;
  double now_batch = static_cast<double>(now_batch_num);
  double decay_batches = static_cast<double>(conf.decay_batches());
  now_batch = std::min(now_batch, decay_batches);
  double cosine_decay = 0.5 * (1.0 + std::cos(PI * now_batch / decay_batches));
  double decayed = (1.0 - conf.alpha()) * cosine_decay + conf.alpha();
  return lr * decayed;
}

double LinearCosineDecayedLearningRate(const LinearCosineDecayConf& conf,
                                       double lr, int64_t now_batch_num) {
  CHECK_GT(conf.decay_batches(), 0);
  const double PI = std::atan(1.0) * 4.0;
  double now_batch = static_cast<double>(now_batch_num);
  double decay_batches = static_cast<double>(conf.decay_batches());
  now_batch = std::min(now_batch, decay_batches);
  double linear_decay = (decay_batches - now_batch) / decay_batches;
  double cosine_decay =
      0.5
      * (1.0
         + std::cos(PI * 2.0 * conf.num_periods() * now_batch / decay_batches));
  double decayed = (conf.alpha() + linear_decay) * cosine_decay + conf.beta();
  return lr * decayed;
}

}  // namespace

double GetDecayedLearningRate(const LearningRateDecayConf& conf, double lr,
                              int64_t now_batch_num) {
  if (conf.has_exponential_conf()) {
    return ExponentialDecayedLearningRate(conf.exponential_conf(), lr,
                                          now_batch_num);
  } else if (conf.has_inverse_time_conf()) {
    return InverseTimeDecayedLearningRate(conf.inverse_time_conf(), lr,
                                          now_batch_num);
  } else if (conf.has_natural_exp_conf()) {
    return NaturalExpDecayedLearningRate(conf.natural_exp_conf(), lr,
                                         now_batch_num);
  } else if (conf.has_piecewise_constant_conf()) {
    return PiecewiseConstantLearningRate(conf.piecewise_constant_conf(), lr,
                                         now_batch_num);
  } else if (conf.has_polynomial_conf()) {
    return PolynomialDecayedLearningRate(conf.polynomial_conf(), lr,
                                         now_batch_num);
  } else if (conf.has_cosine_conf()) {
    return CosineDecayedLearningRate(conf.cosine_conf(), lr, now_batch_num);
  } else if (conf.has_linear_cosine_conf()) {
    return LinearCosineDecayedLearningRate(conf.linear_cosine_conf(), lr,
                                           now_batch_num);
  } else {
    UNIMPLEMENTED();
  }
}

COMMAND(AddKernelCreator(OperatorConf::kNormalMdupdtConf, CreateMdUpdtKernel));

}  // namespace oneflow
