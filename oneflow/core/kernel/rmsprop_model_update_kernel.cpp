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
#include "oneflow/core/kernel/rmsprop_model_update_kernel.h"

namespace oneflow {

namespace {

const RMSPropModelUpdateConf& GetRMSPropModelUpdateConf(const OperatorConf& op_conf) {
  return op_conf.rmsprop_model_update_conf().user_conf().rmsprop_conf();
}

}  // namespace

template<DeviceType device_type, typename T>
const PbMessage& RMSPropMdUpdateKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().rmsprop_model_update_conf();
}

template<DeviceType device_type, typename T>
void RMSPropMdUpdateKernel<device_type, T>::UpdateModel(
    DeviceCtx* ctx, T weight_decay, const int64_t* train_step, const float* learning_rate,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* model_diff_blob = BnInOp2Blob("model_diff");
  Blob* model_blob = BnInOp2Blob("model");
  Blob* mean_square_blob = BnInOp2Blob("mean_square");
  const RMSPropModelUpdateConf& conf = GetRMSPropModelUpdateConf(this->op_conf());

  Blob* mean_gradient_blob = nullptr;
  if (conf.centered()) { mean_gradient_blob = BnInOp2Blob("mean_gradient"); }

  RMSPropMdUpdateKernelUtil<device_type, T>::UpdateModel(
      ctx, model_blob->shape().elem_cnt(), train_step, learning_rate,
      static_cast<T>(conf.decay_rate()), static_cast<T>(conf.epsilon()), conf.centered(),
      weight_decay, model_diff_blob->dptr<T>(), model_blob->mut_dptr<T>(),
      mean_square_blob->mut_dptr<T>(),
      conf.centered() ? mean_gradient_blob->mut_dptr<T>() : nullptr);
}

template<typename T>
class RMSPropMdUpdateKernelUtil<DeviceType::kCPU, T> final {
 public:
  static void UpdateModel(DeviceCtx* ctx, int64_t n, const int64_t* train_step,
                          const float* learning_rate, T decay_rate, T epsilon, bool centered,
                          T weight_decay, const T* model_diff, T* model, T* mean_square,
                          T* mean_gradient) {
    T denom_t;
    for (int64_t i = 0; i < n; ++i) {
      T model_diff_val = model_diff[i];
      mean_square[i] =
          (1 - decay_rate) * model_diff_val * model_diff_val + decay_rate * mean_square[i];
      if (centered) {
        mean_gradient[i] = (1 - decay_rate) * model_diff_val + decay_rate * mean_gradient[i];
        denom_t = mean_square[i] - mean_gradient[i] * mean_gradient[i];
      } else {
        denom_t = mean_square[i];
      }
      model[i] = model[i] - *learning_rate * model_diff_val / std::sqrt(denom_t + epsilon);
    }
  }
};

DEFINE_MDUPDT_KERNEL_CREATOR(RMSProp);

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kRmspropModelUpdateConf, RMSPropMdUpdateKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
