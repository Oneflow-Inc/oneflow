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
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/util/cuda_half_util.h"

namespace oneflow {

namespace {

__global__ void NaiveHalfFillGpu(const int64_t elem_cnt, const float16 x, float16* y) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    y[i] = x;
  }
}

}  // namespace

class HalfConstantLikeKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(HalfConstantLikeKernel);
  HalfConstantLikeKernel() : is_init_(false) {}
  ~HalfConstantLikeKernel() = default;

 private:
  mutable bool is_init_;

  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    if (is_init_) { return; }
    Blob* out_blob = BnInOp2Blob("out");
    float value = 0;
    const auto& conf = this->op_conf().constant_like_conf();
    if (conf.has_int_operand()) {
      value = static_cast<float>(conf.int_operand());
    } else if (conf.has_float_operand()) {
      value = static_cast<float>(conf.float_operand());
    } else {
      UNIMPLEMENTED();
    }
    RUN_CUDA_KERNEL(NaiveHalfFillGpu, ctx.device_ctx, out_blob->static_shape().elem_cnt(),
                    out_blob->static_shape().elem_cnt(), static_cast<float16>(value),
                    out_blob->mut_dptr<float16>());
    
    /*NewKernelUtil<DeviceType::kGPU>::Fill(ctx.device_ctx, out_blob->static_shape().elem_cnt(), static_cast<float16>(value),
                                          out_blob->mut_dptr<float16>());*/
    is_init_ = true;
  }
};

#define REGISTER_HALF_CONSTANT_LIKE_KERNEL \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kConstantLikeConf, DeviceType::kGPU, float16, \
                                        HalfConstantLikeKernel)

REGISTER_HALF_CONSTANT_LIKE_KERNEL

}  // namespace oneflow