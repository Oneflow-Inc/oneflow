#include "oneflow/core/kernel/momentum_model_update_kernel.h"
#include "oneflow/core/kernel/normal_model_update_kernel.cuh"

namespace oneflow {

namespace {

const MomentumModelUpdateConf& GetMomentumModelUpdateConf(const OperatorConf& op_conf) {
  return op_conf.momentum_model_update_conf().user_conf().momentum_conf();
}

}  // namespace

template<DeviceType device_type, typename T>
const PbMessage& MomentumMdUpdateKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().momentum_model_update_conf();
}

template<DeviceType device_type, typename T>
void MomentumMdUpdateKernel<device_type, T>::UpdateModel(
    DeviceCtx* ctx, const T* batch_instance_num_ptr, T learning_rate, T l1, T l2,
    int64_t next_model_vid, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* model_diff_blob = BnInOp2Blob("model_diff");
  Blob* model_blob = BnInOp2Blob("model");
  Blob* momentum_blob = BnInOp2Blob("momentum");
  float beta = GetMomentumModelUpdateConf(this->op_conf()).beta();
  if (next_model_vid == 1) { beta = 0.0f; }

  MomentumMdUpdateKernelUtil<device_type, T>::UpdateModel(
      ctx, model_blob->shape().elem_cnt(), batch_instance_num_ptr, static_cast<T>(beta),
      learning_rate, l1, l2, model_diff_blob->dptr<T>(), model_blob->mut_dptr<T>(),
      momentum_blob->mut_dptr<T>());
}

template<typename T>
class MomentumMdUpdateKernelUtil<DeviceType::kCPU, T> final {
 public:
  static void UpdateModel(DeviceCtx*, int64_t n, const T* batch_instance_num_ptr, T beta,
                          T learning_rate, T l1, T l2, const T* model_diff, T* model, T* momentum) {
    for (int64_t i = 0; i != n; ++i) {
      T reg_diff = RegularizeDiff(model_diff[i], *batch_instance_num_ptr, l1, l2, model[i]);
      momentum[i] = beta * momentum[i] - learning_rate * reg_diff;
      model[i] = model[i] + momentum[i];
    }
  }
};

DEFINE_MDUPDT_KERNEL_CREATOR(Momentum);

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kMomentumModelUpdateConf, MomentumMdUpdateKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
