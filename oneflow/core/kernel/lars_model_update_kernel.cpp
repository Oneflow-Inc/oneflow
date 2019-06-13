#include "oneflow/core/kernel/lars_model_update_kernel.h"
#include "oneflow/core/kernel/normal_model_update_kernel.cuh"

namespace oneflow {

namespace {

const LARSModelUpdateConf& GetLARSModelUpdateConf(const OperatorConf& op_conf) {
  if (Global<JobDesc>::Get()->other_conf().predict_conf().has_tmp_split_fw_bw_train_conf()) {
    return op_conf.lars_model_update_conf().user_conf().lars_conf();
  } else {
    UNIMPLEMENTED();
  }
}

}  // namespace

template<DeviceType device_type, typename T>
const PbMessage& LARSMdUpdateKernel<device_type, T>::GetCustomizedOpConf() const {
  if (Global<JobDesc>::Get()->other_conf().predict_conf().has_tmp_split_fw_bw_train_conf()) {
    return this->op_conf().lars_model_update_conf();
  } else {
    UNIMPLEMENTED();
  }
}

template<DeviceType device_type, typename T>
void LARSMdUpdateKernel<device_type, T>::UpdateModel(
    DeviceCtx* ctx, const T* batch_instance_num_ptr, T learning_rate, T l1, T l2,
    int64_t next_model_vid, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* model_diff_blob = BnInOp2Blob("model_diff");
  Blob* model_blob = BnInOp2Blob("model");
  Blob* momentum_blob = BnInOp2Blob("momentum");
  Blob* data_tmp_blob = BnInOp2Blob("data_tmp");
  const LARSModelUpdateConf& lars_conf = GetLARSModelUpdateConf(this->op_conf());
  if (next_model_vid == 1) {
    Memset<device_type>(ctx, momentum_blob->mut_dptr<T>(), 0,
                        momentum_blob->ByteSizeOfDataContentField());
  }
  Memset<device_type>(ctx, data_tmp_blob->mut_dptr<T>(), 0,
                      data_tmp_blob->ByteSizeOfDataContentField());
  LARSMdUpdateKernelUtil<device_type, T>::UpdateModel(
      ctx, model_blob->shape().elem_cnt(), batch_instance_num_ptr, learning_rate, l1, l2,
      static_cast<T>(lars_conf.momentum_beta()), static_cast<T>(lars_conf.epsilon()),
      static_cast<T>(lars_conf.lars_coefficient()), next_model_vid, model_diff_blob->dptr<T>(),
      model_blob->mut_dptr<T>(), momentum_blob->mut_dptr<T>(), data_tmp_blob->mut_dptr<T>());
}

template<typename T>
class LARSMdUpdateKernelUtil<DeviceType::kCPU, T> final {
 public:
  static void UpdateModel(DeviceCtx* ctx, int64_t n, const T* batch_instance_num_ptr,
                          T learning_rate, T l1, T l2, T momentum_beta, T epsilon,
                          T lars_coefficient, int64_t next_model_vid, const T* model_diff, T* model,
                          T* momentum, T* data_tmp) {
    T model_norm = 0;
    T model_diff_norm = 0;
    FOR_RANGE(int64_t, i, 0, n) {
      model_norm += model[i] * model[i];
      model_diff_norm += model_diff[i] * model_diff[i];
    }
    model_norm = std::sqrt(model_norm / n);
    model_diff_norm = std::sqrt(model_diff_norm / n);
    T local_learning_rate = 0;
    if (next_model_vid == 1) {
      local_learning_rate =
          learning_rate * lars_coefficient * model_norm / (epsilon + model_diff_norm);
    } else {
      local_learning_rate = learning_rate * lars_coefficient * model_norm
                            / (epsilon + model_diff_norm + l2 * model_norm);
    }
    FOR_RANGE(int64_t, i, 0, n) {
      T reg_diff = RegularizeDiff(model_diff[i], *batch_instance_num_ptr, l1, l2, model[i]);
      momentum[i] = momentum_beta * momentum[i] - local_learning_rate * reg_diff;
      model[i] = model[i] + momentum[i];
    }
  }
};

DEFINE_MDUPDT_KERNEL_CREATOR(LARS);

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kLarsModelUpdateConf, LARSMdUpdateKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
