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
#include "oneflow/core/common/scalar.h"
#include "oneflow/core/ep/include/primitive/fill.h"

namespace oneflow {

class ConstantLikeKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConstantLikeKernel);
  ConstantLikeKernel() : is_init_(false) {}
  ~ConstantLikeKernel() = default;

 private:
  mutable bool is_init_;

  void ForwardDataContent(KernelContext* ctx) const override {
    if (is_init_) { return; }
    Blob* out_blob = ctx->BnInOp2Blob("out");
    Scalar value;
    const auto& conf = this->op_conf().constant_like_conf();
    if (conf.has_int_operand()) {
      value = Scalar(conf.int_operand());
    } else if (conf.has_float_operand()) {
      value = Scalar(conf.float_operand());
    } else {
      UNIMPLEMENTED();
    }
    std::unique_ptr<ep::primitive::Fill> primitive =
        ep::primitive::NewPrimitive<ep::primitive::FillFactory>(ctx->stream()->device_type(),
                                                                out_blob->data_type());
    CHECK(primitive);
    primitive->Launch(ctx->stream(), out_blob->mut_dptr(), value,
                      out_blob->static_shape().elem_cnt());
    is_init_ = true;
  }
};

REGISTER_KERNEL(OperatorConf::kConstantLikeConf, ConstantLikeKernel);

}  // namespace oneflow
