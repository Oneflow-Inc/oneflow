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
#ifndef ONEFLOW_XRT_XLA_OPS_OP_KERNEL_H_
#define ONEFLOW_XRT_XLA_OPS_OP_KERNEL_H_

#include "oneflow/xrt/kernel/op_kernel.h"
#include "oneflow/xrt/types.h"
#include "oneflow/xrt/utility/registry.h"
#include "oneflow/xrt/utility/stl.h"
#include "oneflow/xrt/xla/ops/op_context.h"
#include "oneflow/xrt/xla/xla_macro.h"

namespace oneflow {
namespace xrt {
namespace mola {

class XlaOpKernel : public OpKernel<XlaOpContext> {
 public:
  virtual void Compile(XlaOpContext* ctx) = 0;

  XlaOpKernel() = default;
  virtual ~XlaOpKernel() = default;
};

using XlaOpKernelPtr = std::shared_ptr<OpKernel<XlaOpContext>>;

#define REGISTER_XLA_OP_KERNEL(OpName, KernelType)                                            \
  static OpKernelRegistrar<XlaOpContext> _xla_op_kernel_##OpName##_ __attribute__((unused)) = \
      OpKernelRegistrar<XlaOpContext>(#OpName)                                                \
          .SetField(XrtEngine::XLA)                                                           \
          .EnableTrainPhase()                                                                 \
          .SetFactory([]() -> OpKernel<XlaOpContext>* { return new KernelType; })

inline XlaOpKernelPtr BuildOpKernel(const XrtDevice& device, const std::string& op_name) {
  XrtField field = MakeXrtField(device, XrtEngine::XLA);
  return XlaOpKernelPtr(OpKernelBuilder<XlaOpContext>()(field, op_name));
}

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_XLA_OPS_OP_KERNEL_H_
