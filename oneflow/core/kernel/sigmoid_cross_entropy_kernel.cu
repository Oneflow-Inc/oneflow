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
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

namespace {

template<typename PredType, typename LabelType>
__global__ void SigmoidCrossEntropyLossForward(const int64_t n, const PredType* prediction,
                                               const LabelType* label, PredType* loss) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    loss[index] =
        -1.f * prediction[index] * (label[index] - (prediction[index] >= 0))
        + logf(1 + expf(prediction[index] - 2 * prediction[index] * (prediction[index] >= 0)));
  }
}

template<typename PredType, typename LabelType>
__global__ void SigmoidCrossEntropyLossBackward(const int64_t n, const PredType* prediction,
                                                const LabelType* label, PredType* pred_diff) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    pred_diff[index] = 1.f / (1.f + expf(-prediction[index])) - label[index];
  }
}
}  // namespace

template<typename PredType, typename LabelType>
class SigmoidCrossEntropyGpuKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SigmoidCrossEntropyGpuKernel);
  SigmoidCrossEntropyGpuKernel() = default;
  ~SigmoidCrossEntropyGpuKernel() override = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* prediction = BnInOp2Blob("prediction");
    const Blob* label = BnInOp2Blob("label");
    Blob* loss = BnInOp2Blob("loss");
    const int64_t n = prediction->shape().elem_cnt();
    SigmoidCrossEntropyLossForward<PredType, LabelType>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx.device_ctx->cuda_stream()>>>(
            n, prediction->dptr<PredType>(), label->dptr<LabelType>(), loss->mut_dptr<PredType>());
  }
};

template<typename PredType, typename LabelType>
class SigmoidCrossEntropyGradGpuKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SigmoidCrossEntropyGradGpuKernel);
  SigmoidCrossEntropyGradGpuKernel() = default;
  ~SigmoidCrossEntropyGradGpuKernel() override = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* prediction = BnInOp2Blob("prediction");
    const Blob* loss_diff = BnInOp2Blob("loss_diff");
    const Blob* label = BnInOp2Blob("label");
    Blob* pred_diff = BnInOp2Blob("prediction_diff");
    const int64_t n = prediction->shape().elem_cnt();
    SigmoidCrossEntropyLossBackward<PredType>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx.device_ctx->cuda_stream()>>>(
            n, prediction->dptr<PredType>(), label->dptr<LabelType>(),
            pred_diff->mut_dptr<PredType>());
    KernelUtil<DeviceType::kGPU, PredType>::Mul(ctx.device_ctx, n, pred_diff->dptr<PredType>(),
                                                loss_diff->dptr<PredType>(),
                                                pred_diff->mut_dptr<PredType>());
  }
};

#define REGISTER_SIGMOID_CROSS_ENTROPY_GPU_KERNEL(dtype, ltype)                                    \
  NEW_REGISTER_KERNEL(OperatorConf::kSigmoidCrossEntropyConf,                                      \
                      SigmoidCrossEntropyGpuKernel<dtype, ltype>)                                  \
      .SetIsMatchedPred([](const KernelConf& conf) -> bool {                                       \
        return ((conf.op_attribute().op_conf().device_tag() == "gpu")                              \
                && (conf.data_type() == GetDataType<dtype>::value)                                 \
                && (GetDataType<ltype>::value                                                      \
                    == conf.op_attribute().op_conf().sigmoid_cross_entropy_conf().label_type()));  \
      });                                                                                          \
  NEW_REGISTER_KERNEL(OperatorConf::kSigmoidCrossEntropyGradConf,                                  \
                      SigmoidCrossEntropyGradGpuKernel<dtype, ltype>)                              \
      .SetIsMatchedPred([](const KernelConf& conf) -> bool {                                       \
        return (                                                                                   \
            (conf.op_attribute().op_conf().device_tag() == "gpu")                                  \
            && (conf.data_type() == GetDataType<dtype>::value)                                     \
            && (GetDataType<ltype>::value                                                          \
                == conf.op_attribute().op_conf().sigmoid_cross_entropy_grad_conf().label_type())); \
      })

REGISTER_SIGMOID_CROSS_ENTROPY_GPU_KERNEL(float, int32_t);
REGISTER_SIGMOID_CROSS_ENTROPY_GPU_KERNEL(double, int32_t);
REGISTER_SIGMOID_CROSS_ENTROPY_GPU_KERNEL(float, int8_t);
REGISTER_SIGMOID_CROSS_ENTROPY_GPU_KERNEL(double, int8_t);
REGISTER_SIGMOID_CROSS_ENTROPY_GPU_KERNEL(float, float);
REGISTER_SIGMOID_CROSS_ENTROPY_GPU_KERNEL(double, double);

}  // namespace oneflow
