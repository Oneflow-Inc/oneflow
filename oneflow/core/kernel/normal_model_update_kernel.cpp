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
  UpdateModel(ctx.device_ctx, std::get<1>(*tpl),
              DiffAveragingAndL1Regularization(ctx.device_ctx, BnInOp2Blob),
              std::get<0>(*tpl), BnInOp2Blob);
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
      ctx, model->shape().elem_cnt(), l1, model->dptr<T>(),
      in_0->mut_dptr<T>());
  return in_0;
}

template<typename T>
class NormalMdUpdateKernelUtil<DeviceType::kCPU, T> final {
 public:
  static void DiffAveragingAndL1Regularization(DeviceCtx* ctx, int64_t n,
                                               float l1, const T* model,
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

}  // namespace

COMMAND(AddKernelCreator(OperatorConf::kNormalMdupdtConf, CreateMdUpdtKernel));

}  // namespace oneflow
