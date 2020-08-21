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
#include "oneflow/core/kernel/adam_model_update_kernel.h"

namespace oneflow {

namespace {

const AdamModelUpdateConf& GetAdamModelUpdateConf(const OperatorConf& op_conf) {
  return op_conf.adam_model_update_conf().user_conf().adam_conf();
};

template<typename T>
void UpdateMomentEstimate(int64_t n, T beta, int32_t p, const T* model_diff, T* moment) {
  FOR_RANGE(int64_t, i, 0, n) {
    // Update biased moment estimate
    moment[i] = beta * moment[i] + (1 - beta) * std::pow(model_diff[i], p);
  }
}

}  // namespace

template<DeviceType device_type, typename T>
const PbMessage& AdamMdUpdateKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().adam_model_update_conf();
}

template<DeviceType device_type, typename T>
void AdamMdUpdateKernel<device_type, T>::UpdateModel(
    DeviceCtx* ctx, T weight_decay, const int64_t* train_step, const float* learning_rate,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* model_blob = BnInOp2Blob("model");
  Blob* m_blob = BnInOp2Blob("m");
  Blob* v_blob = BnInOp2Blob("v");
  Blob* beta1_t_blob = BnInOp2Blob("beta1_t");
  Blob* beta2_t_blob = BnInOp2Blob("beta2_t");
  const auto& adam_conf = GetAdamModelUpdateConf(this->op_conf());
  AdamMdUpdateKernelUtil<device_type, T>::UpdateModel(
      ctx, model_blob->shape().elem_cnt(), learning_rate, weight_decay,
      static_cast<T>(adam_conf.beta1()), static_cast<T>(adam_conf.beta2()),
      static_cast<T>(adam_conf.epsilon()), adam_conf.do_bias_correction(), train_step,
      (beta1_t_blob ? beta1_t_blob->mut_dptr<T>() : nullptr),
      (beta2_t_blob ? beta2_t_blob->mut_dptr<T>() : nullptr), BnInOp2Blob("model_diff")->dptr<T>(),
      model_blob->mut_dptr<T>(), m_blob->mut_dptr<T>(), v_blob->mut_dptr<T>());
}

template<typename T>
class AdamMdUpdateKernelUtil<DeviceType::kCPU, T> final {
 public:
  static void UpdateModel(DeviceCtx* ctx, int64_t n, const float* learning_rate, T weight_decay,
                          T beta1, T beta2, T epsilon, bool do_bias_correction,
                          const int64_t* train_step, T* beta1_t, T* beta2_t, const T* model_diff,
                          T* model, T* m, T* v) {
    float lr;
    if (do_bias_correction) {
      lr = *learning_rate * std::sqrt(1 - *beta2_t) / (1 - *beta1_t);
      *beta1_t *= beta1;
      *beta2_t *= beta2;
    } else {
      lr = *learning_rate;
    }
    UpdateMomentEstimate<T>(n, beta1, 1, model_diff, m);
    // second-order moment
    UpdateMomentEstimate<T>(n, beta2, 2, model_diff, v);
    FOR_RANGE(int64_t, i, 0, n) {
      const T mdv = m[i] / (std::sqrt(v[i]) + epsilon);
      model[i] = model[i] - lr * (mdv + weight_decay * model[i]);
    }
  }
};

DEFINE_MDUPDT_KERNEL_CREATOR(Adam);

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kAdamModelUpdateConf, AdamMdUpdateKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
