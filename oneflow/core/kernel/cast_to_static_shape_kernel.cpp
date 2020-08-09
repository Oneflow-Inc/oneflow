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
#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

template<DeviceType device_type>
class CastToStaticShapeKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CastToStaticShapeKernel);
  CastToStaticShapeKernel() = default;
  ~CastToStaticShapeKernel() override = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in = BnInOp2Blob("in");
    Blob* out = BnInOp2Blob("out");
    CHECK(in->shape() == ShapeView(in->static_shape()));
    CHECK_EQ(out->shape(), in->shape());
    out->CopyValidDataContentFrom(ctx.device_ctx, in);
  }
};

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kCastToStaticShapeConf, CastToStaticShapeKernel);

}  // namespace oneflow
