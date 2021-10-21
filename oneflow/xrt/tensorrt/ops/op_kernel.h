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
#ifndef ONEFLOW_XRT_TENSORRT_OPS_OP_KERNEL_H_
#define ONEFLOW_XRT_TENSORRT_OPS_OP_KERNEL_H_

#include "oneflow/xrt/kernel/op_kernel.h"
#include "oneflow/xrt/tensorrt/ops/op_context.h"
#include "oneflow/xrt/types.h"
#include "oneflow/xrt/utility/registry.h"
#include "oneflow/xrt/utility/stl.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

class TrtOpKernel : public OpKernel<TrtOpContext> {
 public:
  virtual void Compile(TrtOpContext* ctx) = 0;

  TrtOpKernel() = default;
  virtual ~TrtOpKernel() = default;
};

using TrtOpKernelPtr = std::shared_ptr<OpKernel<TrtOpContext>>;

#define REGISTER_TRT_OP_KERNEL(OpName, KernelType)                                            \
  static OpKernelRegistrar<TrtOpContext> _trt_op_kernel_##OpName##_ __attribute__((unused)) = \
      OpKernelRegistrar<TrtOpContext>(#OpName)                                                \
          .SetField(XrtEngine::TENSORRT)                                                      \
          .SetDevice({XrtDevice::GPU_CUDA})                                                   \
          .SetFactory([]() -> OpKernel<TrtOpContext>* { return new KernelType; })

inline TrtOpKernelPtr BuildOpKernel(const std::string& op_name) {
  auto field = MakeXrtField(XrtDevice::GPU_CUDA, XrtEngine::TENSORRT);
  return TrtOpKernelPtr(OpKernelBuilder<TrtOpContext>()(field, op_name));
}

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_TENSORRT_OPS_OP_KERNEL_H_
