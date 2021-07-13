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
#ifndef ONEFLOW_XRT_TVM_OPS_OP_KERNEL_H_
#define ONEFLOW_XRT_TVM_OPS_OP_KERNEL_H_

#include "oneflow/xrt/kernel/op_kernel.h"
#include "oneflow/xrt/types.h"
#include "oneflow/xrt/utility/registry.h"
#include "oneflow/xrt/utility/stl.h"
#include "oneflow/xrt/tvm/ops/op_context.h"

namespace oneflow {
namespace xrt {
namespace of_tvm {

class TVMOpKernel : public OpKernel<TVMOpContext> {
 public:
  virtual void Compile(TVMOpContext* ctx) = 0;

  TVMOpKernel() = default;
  virtual ~TVMOpKernel() = default;
};

// TODO: add tvm cpu support
#define REGISTER_TVM_OP_KERNEL(OpName, KernelType)                                            \
  static OpKernelRegistrar<TVMOpContext> _tvm_op_kernel_##OpName##_ __attribute__((unused)) = \
      OpKernelRegistrar<TVMOpContext>(#OpName)                                                \
          .SetField(XrtEngine::TVM)                                                           \
          .SetDevice({XrtDevice::GPU_CUDA})                                                   \
          .SetFactory([]() -> OpKernel<TVMOpContext>* { return new KernelType; })

using TVMOpKernelPtr = std::shared_ptr<OpKernel<TVMOpContext>>;

inline TVMOpKernelPtr BuildTVMOpKernel(const std::string& op_name) {
  auto field = MakeXrtField(XrtDevice::GPU_CUDA, XrtEngine::TVM);
  return TVMOpKernelPtr(OpKernelBuilder<TVMOpContext>()(field, op_name));
}

}  // namespace of_tvm
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_TVM_OPS_OP_KERNEL_H_