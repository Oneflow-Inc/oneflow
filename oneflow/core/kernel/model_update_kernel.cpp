#include "oneflow/core/kernel/model_update_kernel.h"
#include "oneflow/core/kernel/normal_model_update_kernel.h"
#include "oneflow/core/kernel/momentum_model_update_kernel.h"
#include "oneflow/core/kernel/rmsprop_model_update_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void MdUpdateKernel<device_type, T>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto tpl = reinterpret_cast<std::tuple<int64_t, const Blob*>*>(ctx.other);
  UpdateModel(ctx.device_ctx, std::get<1>(*tpl),
              DiffAveragingAndRegularization(ctx.device_ctx, BnInOp2Blob),
              std::get<0>(*tpl), BnInOp2Blob);
}

template<DeviceType device_type, typename T>
Blob* MdUpdateKernel<device_type, T>::DiffAveragingAndRegularization(
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
  float l1 = JobDesc::Singleton()->L1();
  float l2 = JobDesc::Singleton()->L2();
  MdUpdateKernelUtil<device_type, T>::DiffAveragingAndRegularization(
      ctx, model->shape().elem_cnt(), l1, l2, model->dptr<T>(),
      in_0->mut_dptr<T>());
  return in_0;
}

template<typename T>
class MdUpdateKernelUtil<DeviceType::kCPU, T> final {
 public:
  static void DiffAveragingAndRegularization(DeviceCtx* ctx, int64_t n,
                                             float l1, float l2, const T* model,
                                             T* model_diff_acc) {
    T zero = static_cast<T>(0);
    for (int64_t i = 0; i != n; ++i) {
      model_diff_acc[i] /= JobDesc::Singleton()->BatchSize();
      model_diff_acc[i] +=
          l1 * ((model[i] >= zero) - (model[i] <= zero)) + l2 * model[i];
    }
  }
};

#define INSTANTIATE_KERNEL_UTIL(data_type_pair)       \
  template class MdUpdateKernelUtil<DeviceType::kCPU, \
                                    OF_PP_PAIR_FIRST(data_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_KERNEL_UTIL,
                                 FLOATING_DATA_TYPE_SEQ)

#define INSTANTIATE_KERNEL(device_type, data_type_pair) \
  template class MdUpdateKernel<device_type, OF_PP_PAIR_FIRST(data_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_KERNEL, DEVICE_TYPE_SEQ,
                                 FLOATING_DATA_TYPE_SEQ)

namespace {

Kernel* CreateMdUpdtKernel(const KernelConf& kernel_conf) {
  const ModelUpdateOpUserConf& user_conf =
      kernel_conf.op_conf().mdupdt_conf().user_conf();
  if (user_conf.has_normal_conf()) {
    return CreateNormalMdUpdtKernel(kernel_conf);
  } else if (user_conf.has_momentum_conf()) {
    return CreateMomentumMdUpdtKernel(kernel_conf);
  } else if (user_conf.has_rmsprop_conf()) {
    return CreateRMSPropMdUpdtKernel(kernel_conf);
  } else {
    UNEXPECTED_RUN();
  }
}

}  // namespace

COMMAND(AddKernelCreator(OperatorConf::kMdupdtConf, CreateMdUpdtKernel));

}  // namespace oneflow
