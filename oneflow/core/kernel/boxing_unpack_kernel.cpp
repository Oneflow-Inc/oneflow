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
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class BoxingUnpackKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingUnpackKernel);
  BoxingUnpackKernel() = default;
  ~BoxingUnpackKernel() override = default;

 private:
  bool IsStateless() const override { return false; }
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
void BoxingUnpackKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in = BnInOp2Blob("in");
  Blob* out = BnInOp2Blob("out");
  const BoxingUnpackOpConf& boxing_unpack_conf = this->op_conf().boxing_unpack_conf();
  if (boxing_unpack_conf.need_transpose()) {
    const Shape transpose_in_shape(boxing_unpack_conf.transpose_in_shape());
    const Shape transpose_out_shape(boxing_unpack_conf.transpose_out_shape());
    NewKernelUtil<device_type>::Transpose(
        ctx.device_ctx, transpose_in_shape.NumAxes(), transpose_in_shape, transpose_out_shape, boxing_unpack_conf.transpose_perm(),
        transpose_in_shape.elem_cnt(), in->dptr<T>(), out->mut_dptr<T>());
  } else {
    out->CopyDataContentFrom(ctx.device_ctx, in);
  }
}

#ifdef WITH_CUDA
#define REGISTER_BOXING_UNPACK_KERNEL(dtype)                                                    \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kBoxingUnpackConf, DeviceType::kGPU, dtype, \
                                        BoxingUnpackKernel<DeviceType::kGPU, dtype>)              
REGISTER_BOXING_UNPACK_KERNEL(float);
REGISTER_BOXING_UNPACK_KERNEL(double);
REGISTER_BOXING_UNPACK_KERNEL(int8_t);
REGISTER_BOXING_UNPACK_KERNEL(int32_t);
REGISTER_BOXING_UNPACK_KERNEL(int64_t);
#endif

}  // namespace oneflow
