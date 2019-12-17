#include "oneflow/core/kernel/normal_model_update_kernel.h"
#include "oneflow/core/kernel/naive_model_update_kernel.h"
#include "oneflow/core/kernel/momentum_model_update_kernel.h"
#include "oneflow/core/kernel/rmsprop_model_update_kernel.h"
#include "oneflow/core/kernel/lars_model_update_kernel.h"
#include "oneflow/core/kernel/adam_model_update_kernel.h"
#include "oneflow/core/kernel/lazy_adam_model_update_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void NormalMdUpdateKernel<device_type, T>::VirtualKernelInit() {
  const PbMessage& op_conf = this->GetCustomizedOpConf();
  user_conf_ = *GetMsgPtrFromPbMessage<NormalModelUpdateOpUserConf>(op_conf, "user_conf");
  l1_ = static_cast<T>(GetValFromPbMessage<float>(op_conf, "l1"));
  l2_ = static_cast<T>(GetValFromPbMessage<float>(op_conf, "l2"));
}

template<DeviceType device_type, typename T>
void NormalMdUpdateKernel<device_type, T>::Forward(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const T* batch_instance_num_ptr = BnInOp2Blob("total_instance_num_diff")->dptr<T>();
  const int64_t* train_step_ptr = BnInOp2Blob("train_step")->dptr<int64_t>();
  const float* learning_rate_ptr = BnInOp2Blob("learning_rate")->dptr<float>();
  if (user_conf_.has_clip_conf()) {
    ClipGradient(ctx.device_ctx, user_conf_.clip_conf(), batch_instance_num_ptr, BnInOp2Blob);
  }
  UpdateModel(ctx.device_ctx, batch_instance_num_ptr, l1_, l2_, train_step_ptr, learning_rate_ptr,
              BnInOp2Blob);
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

template<DeviceType device_type, typename T>
void ClipByGlobalNorm(DeviceCtx* ctx, const ClipByGlobalNormConf& conf,
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
void NormalMdUpdateKernel<device_type, T>::ClipGradient(
    DeviceCtx* ctx, const ClipConf& conf, const T* batch_instance_num_ptr,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (conf.has_clip_by_global_norm()) {
    ClipByGlobalNorm<device_type, T>(ctx, conf.clip_by_global_norm(), batch_instance_num_ptr,
                                     BnInOp2Blob);
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
