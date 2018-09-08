#include "oneflow/core/kernel/momentum_model_update_kernel.h"
#include "oneflow/core/kernel/normal_model_update_kernel.cuh"

namespace oneflow {

template<DeviceType device_type, typename T>
void MomentumMdUpdateKernel<device_type, T>::UpdateModel(
    DeviceCtx* ctx, int64_t batch_size, T learning_rate, T l1, T l2, const Blob* pre_model_blob,
    int64_t next_model_vid, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* model_diff_blob = BnInOp2Blob("model_diff");
  Blob* model_blob = BnInOp2Blob("model");
  Blob* momentum_blob = BnInOp2Blob("momentum");
  float beta = this->op_conf().normal_mdupdt_conf().user_conf().momentum_conf().beta();
  if (next_model_vid == 1) { beta = 0.0f; }

  MomentumMdUpdateKernelUtil<device_type, T>::UpdateModel(
      ctx, model_blob->shape().elem_cnt(), batch_size, static_cast<T>(beta), learning_rate, l1, l2,
      model_diff_blob->dptr<T>(), pre_model_blob->dptr<T>(), momentum_blob->mut_dptr<T>(),
      model_blob->mut_dptr<T>());
}

template<typename T>
class MomentumMdUpdateKernelUtil<DeviceType::kCPU, T> final {
 public:
  static void UpdateModel(DeviceCtx*, int64_t n, int64_t batch_size, T beta, T learning_rate, T l1,
                          T l2, const T* model_diff, const T* pre_model, T* momentum, T* model) {
    for (int64_t i = 0; i != n; ++i) {
      T reg_diff = RegularizeDiff(model_diff[i], batch_size, l1, l2, pre_model[i]);
      momentum[i] = beta * momentum[i] - learning_rate * reg_diff;
      model[i] = pre_model[i] + momentum[i];
    }
  }
};

DEFINE_MDUPDT_KERNEL_CREATOR(Momentum);

}  // namespace oneflow
