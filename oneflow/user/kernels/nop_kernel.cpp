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
#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

class NopKernel final : public user_op::OpKernel {
 public:
  NopKernel() = default;
  ~NopKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    // do nothing
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_NOP_KERNEL(op_type_name) \
  REGISTER_USER_KERNEL(op_type_name).SetCreateFn<NopKernel>().SetIsMatchedHob(user_op::HobTrue());

REGISTER_NOP_KERNEL("cast_to_tick")

}  // namespace

}  // namespace oneflow
