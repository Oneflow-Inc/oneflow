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
#include "oneflow/core/kernel/momentum_model_update_kernel.h"

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
    DeviceCtx* ctx, T weight_decay, const int64_t* train_step, const float* learning_rate,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* model_diff_blob = BnInOp2Blob("model_diff");
  Blob* model_blob = BnInOp2Blob("model");
  Blob* momentum_blob = BnInOp2Blob("momentum");
  float beta = GetMomentumModelUpdateConf(this->op_conf()).beta();

  MomentumMdUpdateKernelUtil<device_type, T>::UpdateModel(
      ctx, model_blob->shape().elem_cnt(), static_cast<T>(beta), train_step, learning_rate,
      weight_decay, model_diff_blob->dptr<T>(), model_blob->mut_dptr<T>(),
      momentum_blob->mut_dptr<T>());
}

template<typename T>
class MomentumMdUpdateKernelUtil<DeviceType::kCPU, T> final {
 public:
  static void UpdateModel(DeviceCtx*, int64_t n, T beta, const int64_t* train_step,
                          const float* learning_rate, T weight_decay, const T* model_diff, T* model,
                          T* momentum) {
    for (int64_t i = 0; i != n; ++i) {
      T next_momentum = beta * momentum[i] - *learning_rate * model_diff[i];
      momentum[i] = next_momentum;
      model[i] = model[i] + next_momentum - *learning_rate * weight_decay * model[i];
    }
  }
};

DEFINE_MDUPDT_KERNEL_CREATOR(Momentum);

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kMomentumModelUpdateConf, MomentumMdUpdateKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
