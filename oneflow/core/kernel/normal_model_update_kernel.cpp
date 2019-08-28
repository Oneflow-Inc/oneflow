#include "oneflow/core/kernel/normal_model_update_kernel.h"
#include "oneflow/core/kernel/naive_model_update_kernel.h"
#include "oneflow/core/kernel/momentum_model_update_kernel.h"
#include "oneflow/core/kernel/rmsprop_model_update_kernel.h"
#include "oneflow/core/kernel/lars_model_update_kernel.h"
#include "oneflow/core/kernel/adam_model_update_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void NormalMdUpdateKernel<device_type, T>::Forward(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  int64_t cur_batch_num = std::get<0>(
      *(reinterpret_cast<std::tuple<int64_t, std::function<const Blob*(const LogicalBlobId&)>>*>(
          ctx.other)));
  int64_t next_model_vid = cur_batch_num + 1;
  const PbMessage& op_conf = this->GetCustomizedOpConf();
  const auto& conf = *GetMsgPtrFromPbMessage<NormalModelUpdateOpUserConf>(op_conf, "user_conf");
  float learning_rate = GetValFromPbMessage<float>(op_conf, "learning_rate");
  const T* batch_instance_num_ptr = BnInOp2Blob("total_instance_num_diff")->dptr<T>();
  if (conf.has_clip_conf()) {
    ClipGradient(ctx.device_ctx, cur_batch_num, conf.clip_conf(), batch_instance_num_ptr,
                 BnInOp2Blob);
  }
  if (TriggerWarmup(conf, learning_rate, next_model_vid)) {
    learning_rate = GetWarmupLearningRate(conf.warmup_conf(), learning_rate, next_model_vid);
  } else if (conf.has_learning_rate_decay()) {
    learning_rate =
        GetDecayedLearningRate(conf.learning_rate_decay(), learning_rate, cur_batch_num);
  }
  float l1 = GetValFromPbMessage<float>(op_conf, "l1");
  float l2 = GetValFromPbMessage<float>(op_conf, "l2");
  UpdateModel(ctx.device_ctx, batch_instance_num_ptr, static_cast<T>(learning_rate),
              static_cast<T>(l1), static_cast<T>(l2), next_model_vid, BnInOp2Blob);
}

#define INSTANTIATE_KERNEL(device_type, data_type_pair) \
  template struct NormalMdUpdateKernel<device_type, OF_PP_PAIR_FIRST(data_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_KERNEL, DEVICE_TYPE_SEQ, FLOATING_DATA_TYPE_SEQ)

namespace {

Kernel* CreateMdUpdtKernel(const KernelConf& kernel_conf) {
  const NormalModelUpdateOpUserConf& user_conf =
      kernel_conf.op_attribute().op_conf().normal_mdupdt_conf().user_conf();
  if (user_conf.has_naive_conf()) {
    return CreateNaiveMdUpdtKernel(kernel_conf);
  } else if (user_conf.has_momentum_conf()) {
    return CreateMomentumMdUpdtKernel(kernel_conf);
  } else if (user_conf.has_rmsprop_conf()) {
    return CreateRMSPropMdUpdtKernel(kernel_conf);
  } else if (user_conf.has_lars_conf()) {
    return CreateLARSMdUpdtKernel(kernel_conf);
  } else if (user_conf.has_adam_conf()) {
    return CreateAdamMdUpdtKernel(kernel_conf);
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

double ConstantWarmupLearningRate(const ConstantWarmupConf& conf, double lr,
                                  int64_t next_batch_num) {
  CHECK_GT(conf.warmup_batches(), 0);
  CHECK_GT(conf.multiplier(), 0);
  CHECK_LT(conf.multiplier(), 1);
  if (next_batch_num <= conf.warmup_batches()) {
    return lr * conf.multiplier();
  } else {
    return lr;
  }
}

double LinearWarmupLearningRate(const LinearWarmupConf& conf, double lr, int64_t next_batch_num) {
  CHECK_GT(conf.warmup_batches(), 0);
  CHECK_GE(conf.start_multiplier(), 0);
  CHECK_LT(conf.start_multiplier(), 1);
  double start_multiplier = conf.start_multiplier();
  double multiplier = 1.0;
  if (next_batch_num <= conf.warmup_batches()) {
    multiplier =
        start_multiplier + (1.0 - start_multiplier) * (next_batch_num * 1.0 / conf.warmup_batches());
  }
  return lr * multiplier;
}

template<DeviceType device_type, typename T>
void ClipByGlobalNorm(DeviceCtx* ctx, const int64_t cur_batch_num, const ClipByGlobalNormConf& conf,
                      const T* batch_instance_num_ptr,
                      std::function<Blob*(const std::string&)> BnInOp2Blob) {
  int64_t n = BnInOp2Blob("model_diff")->shape().elem_cnt();
  T* model_diff = BnInOp2Blob("model_diff")->mut_dptr<T>();
  T* global_norm_ptr = BnInOp2Blob("data_tmp")->mut_dptr<T>();
  if (conf.has_global_norm()) {
    KernelUtil<device_type, T>::Set(ctx, static_cast<T>(conf.global_norm()), global_norm_ptr);
  } else {
    // The Dot does not read the result, so the global_norm need not be initialized.
    KernelUtil<device_type, T>::Dot(ctx, n, model_diff, 1, model_diff, 1, global_norm_ptr);
    KernelUtil<device_type, T>::Sqrt(ctx, 1, global_norm_ptr, global_norm_ptr);
    KernelUtil<device_type, T>::Div(ctx, 1, global_norm_ptr, batch_instance_num_ptr);
  }
  T* ratio_ptr = BnInOp2Blob("data_tmp")->mut_dptr<T>();
  NormalMdUpdateKernelUtil<device_type, T>::CmptClipRatioByGlobalNorm(
      ctx, global_norm_ptr, static_cast<T>(conf.clip_norm()), ratio_ptr);
  KernelUtil<device_type, T>::Scal(ctx, n, ratio_ptr, model_diff, 1);
}

}  // namespace

template<DeviceType device_type, typename T>
bool NormalMdUpdateKernel<device_type, T>::TriggerWarmup(const NormalModelUpdateOpUserConf& conf,
                                                         double lr, int64_t next_batch_num) const {
  if (!conf.has_warmup_conf()) { return false; }
  const WarmupConf& warmup_conf = conf.warmup_conf();
  if (warmup_conf.has_constant_conf()) {
    return (next_batch_num <= warmup_conf.constant_conf().warmup_batches());
  } else if (warmup_conf.has_linear_conf()) {
    return (next_batch_num <= warmup_conf.linear_conf().warmup_batches());
  } else {
    UNIMPLEMENTED();
  }
}

template<DeviceType device_type, typename T>
double NormalMdUpdateKernel<device_type, T>::GetWarmupLearningRate(const WarmupConf& conf,
                                                                   double lr,
                                                                   int64_t next_batch_num) const {
  if (conf.has_constant_conf()) {
    return ConstantWarmupLearningRate(conf.constant_conf(), lr, next_batch_num);
  } else if (conf.has_linear_conf()) {
    return LinearWarmupLearningRate(conf.linear_conf(), lr, next_batch_num);
  } else {
    UNIMPLEMENTED();
  }
}

template<DeviceType device_type, typename T>
void NormalMdUpdateKernel<device_type, T>::ClipGradient(
    DeviceCtx* ctx, const int64_t cur_batch_num, const ClipConf& conf,
    const T* batch_instance_num_ptr, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (conf.has_clip_by_global_norm()) {
    ClipByGlobalNorm<device_type, T>(ctx, cur_batch_num, conf.clip_by_global_norm(),
                                     batch_instance_num_ptr, BnInOp2Blob);
  } else {
    UNIMPLEMENTED();
  }
}

template<DeviceType device_type, typename T>
double NormalMdUpdateKernel<device_type, T>::GetDecayedLearningRate(
    const LearningRateDecayConf& conf, double lr, int64_t cur_batch_num) const {
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
  } else if (conf.has_linear_cosine_conf()) {
    return LinearCosineDecayedLearningRate(conf.linear_cosine_conf(), lr, cur_batch_num);
  } else {
    UNIMPLEMENTED();
  }
}

template<typename T>
class NormalMdUpdateKernelUtil<DeviceType::kCPU, T> final {
 public:
  static void CmptClipRatioByGlobalNorm(DeviceCtx* ctx, const T* global_norm_ptr, T clip_norm,
                                        T* ratio_ptr) {
    *ratio_ptr = clip_norm / std::max(*global_norm_ptr, clip_norm);
  }
};

REGISTER_KERNEL_CREATOR(OperatorConf::kNormalMdupdtConf, CreateMdUpdtKernel);

}  // namespace oneflow
