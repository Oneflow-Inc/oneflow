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
#ifndef ONEFLOW_XRT_OPENVINO_OPS_OP_KERNEL_H_
#define ONEFLOW_XRT_OPENVINO_OPS_OP_KERNEL_H_

#include "oneflow/xrt/kernel/op_kernel.h"
#include "oneflow/xrt/openvino/ops/op_context.h"
#include "oneflow/xrt/types.h"
#include "oneflow/xrt/utility/registry.h"
#include "oneflow/xrt/utility/stl.h"

namespace oneflow {
namespace xrt {
namespace openvino {

class OpenvinoOpKernel : public OpKernel<OpenvinoOpContext> {
 public:
  virtual void Compile(OpenvinoOpContext* ctx) = 0;

  OpenvinoOpKernel() = default;
  virtual ~OpenvinoOpKernel() = default;
};

using OpenvinoOpKernelPtr = std::shared_ptr<OpKernel<OpenvinoOpContext>>;

#define REGISTER_OPENVINO_OP_KERNEL(OpName, KernelType)                       \
  static OpKernelRegistrar<OpenvinoOpContext> _openvino_op_kernel_##OpName##_ \
      __attribute__((unused)) =                                               \
          OpKernelRegistrar<OpenvinoOpContext>(#OpName)                       \
              .SetField(XrtEngine::OPENVINO)                                  \
              .SetDevice({XrtDevice::CPU_X86})                                \
              .SetFactory([]() -> OpKernel<OpenvinoOpContext>* { return new KernelType; })

inline OpenvinoOpKernelPtr BuildOpKernel(const std::string& op_name) {
  auto field = MakeXrtField(XrtDevice::CPU_X86, XrtEngine::OPENVINO);
  return OpenvinoOpKernelPtr(OpKernelBuilder<OpenvinoOpContext>()(field, op_name));
}

}  // namespace openvino
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_OPENVINO_OPS_OP_KERNEL_H_
