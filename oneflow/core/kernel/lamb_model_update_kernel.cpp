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
#include "oneflow/core/kernel/lamb_model_update_kernel.h"

namespace oneflow {

namespace {

const LAMBModelUpdateConf& GetLAMBModelUpdateConf(const OperatorConf& op_conf) {
  return op_conf.lamb_model_update_conf().user_conf().lamb_conf();
};

template<typename T>
void UpdateMomentEstimate(int64_t n, T beta, int32_t p, const T* model_diff, T* moment) {
  FOR_RANGE(int64_t, i, 0, n) {
    moment[i] = beta * moment[i] + (1 - beta) * std::pow(model_diff[i], p);
  }
}

}  // namespace

template<DeviceType device_type, typename T>
const PbMessage& LAMBMdUpdateKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().lamb_model_update_conf();
}

template<DeviceType device_type, typename T>
void LAMBMdUpdateKernel<device_type, T>::UpdateModel(
    DeviceCtx* ctx, T weight_decay, const int64_t* train_step, const float* learning_rate,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* model_blob = BnInOp2Blob("model");
  Blob* m_blob = BnInOp2Blob("m");
  Blob* v_blob = BnInOp2Blob("v");
  Blob* beta1_t_blob = BnInOp2Blob("beta1_t");
  Blob* beta2_t_blob = BnInOp2Blob("beta2_t");
  Blob* fw_buf_blob = BnInOp2Blob("fw_buf");
  const auto& lamb_conf = GetLAMBModelUpdateConf(this->op_conf());
  if (*train_step != 0) {
    *beta1_t_blob->mut_dptr<T>() *= lamb_conf.beta1();
    *beta2_t_blob->mut_dptr<T>() *= lamb_conf.beta2();
  }
  Memset<device_type>(ctx, fw_buf_blob->mut_dptr<T>(), 0, fw_buf_blob->ByteSizeOfBlobBody());
  LAMBMdUpdateKernelUtil<device_type, T>::UpdateModel(
      ctx, model_blob->shape().elem_cnt(), learning_rate, weight_decay,
      static_cast<T>(lamb_conf.beta1()), static_cast<T>(lamb_conf.beta2()),
      static_cast<T>(lamb_conf.epsilon()), train_step,
      (beta1_t_blob ? beta1_t_blob->dptr<T>() : nullptr),
      (beta2_t_blob ? beta2_t_blob->dptr<T>() : nullptr), BnInOp2Blob("model_diff")->mut_dptr<T>(),
      model_blob->mut_dptr<T>(), m_blob->mut_dptr<T>(), v_blob->mut_dptr<T>(),
      fw_buf_blob->mut_dptr<T>());
}

template<typename T>
class LAMBMdUpdateKernelUtil<DeviceType::kCPU, T> final {
 public:
  static void UpdateModel(DeviceCtx* ctx, int64_t n, const float* learning_rate, T weight_decay,
                          T beta1, T beta2, T epsilon, const int64_t* train_step, const T* beta1_t,
                          const T* beta2_t, T* model_diff, T* model, T* m, T* v, T* fw_buf) {
    // first-order moment
    UpdateMomentEstimate<T>(n, beta1, 1, model_diff, m);
    // second-order moment
    UpdateMomentEstimate<T>(n, beta2, 2, model_diff, v);
    FOR_RANGE(int64_t, i, 0, n) {
      model_diff[i] = (m[i] / (1 - *beta1_t)) / std::sqrt(v[i] / (1 - *beta2_t) + epsilon);
    }
    KernelUtil<DeviceType::kCPU, T>::Dot(ctx, n, model, 1, model, 1, &fw_buf[0]);
    KernelUtil<DeviceType::kCPU, T>::Dot(ctx, n, model_diff, 1, model_diff, 1, &fw_buf[1]);
    KernelUtil<DeviceType::kCPU, T>::Sqrt(ctx, 2, fw_buf, fw_buf);
    const float local_lr = fw_buf[0] / fw_buf[1] * *learning_rate;
    FOR_RANGE(int64_t, i, 0, n) {
      model[i] = model[i] - local_lr * (model_diff[i] + weight_decay * model[i]);
    }
  }
};

DEFINE_MDUPDT_KERNEL_CREATOR(LAMB);

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kLambModelUpdateConf, LAMBMdUpdateKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
