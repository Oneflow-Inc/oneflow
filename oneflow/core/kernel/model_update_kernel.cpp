#include "oneflow/core/kernel/model_update_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void MdUpdateKernel<device_type, T>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto tpl = reinterpret_cast<std::tuple<int64_t, const Blob*>*>(ctx.other);
  Regularization(ctx.device_ctx, BnInOp2Blob);
  UpdateModel(ctx.device_ctx, std::get<1>(*tpl), std::get<0>(*tpl),
              BnInOp2Blob);
}

template<DeviceType device_type, typename T>
void MdUpdateKernel<device_type, T>::Regularization(
    DeviceCtx* ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* model_diff_acc_blob = BnInOp2Blob("model_diff_acc");
  const Blob* model_blob = BnInOp2Blob("model");
  float l1 = JobDesc::Singleton()->L1();
  float l2 = JobDesc::Singleton()->L2();
  if (l1 != 0.f && l2 != 0.f) {
    MdUpdateKernelUtil<device_type, T>::L1L2Regularization(
        ctx, model_blob->shape().elem_cnt(), l1, l2, model_blob->dptr<T>(),
        model_diff_acc_blob->mut_dptr<T>());
  } else if (l1 != 0.f) {
    MdUpdateKernelUtil<device_type, T>::L1Regularization(
        ctx, model_blob->shape().elem_cnt(), l1, model_blob->dptr<T>(),
        model_diff_acc_blob->mut_dptr<T>());
  } else if (l2 != 0.f) {
    KernelUtil<device_type, T>::Axpy(
        ctx, model_diff_acc_blob->shape().elem_cnt(), static_cast<T>(l2),
        model_blob->dptr<T>(), 1, model_diff_acc_blob->mut_dptr<T>(), 1);
  } else {
    // do nothing
  }
}

template<typename T>
class MdUpdateKernelUtil<DeviceType::kCPU, T> final {
 public:
  static void L1Regularization(DeviceCtx* ctx, int64_t n, float l1,
                               const T* model, T* model_diff_acc) {
    T zero = static_cast<T>(0);
    for (int64_t i = 0; i != n; ++i) {
      model_diff_acc[i] += l1 * ((model[i] > zero) - (zero < model[i]));
    }
  }
  static void L1L2Regularization(DeviceCtx* ctx, int64_t n, float l1, float l2,
                                 const T* model, T* model_diff_acc) {
    T zero = static_cast<T>(0);
    for (int64_t i = 0; i != n; ++i) {
      model_diff_acc[i] +=
          l1 * ((model[i] > zero) - (zero < model[i])) + l2 * model[i];
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

}  // namespace oneflow
