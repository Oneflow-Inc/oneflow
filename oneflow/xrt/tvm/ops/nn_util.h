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
#ifndef ONEFLOW_XRT_TVM_OPS_NN_UTIL_H_
#define ONEFLOW_XRT_TVM_OPS_NN_UTIL_H_

#include "oneflow/xrt/tvm/ops/op_kernel.h"
#include <tvm/relay/attrs/nn.h>
#include <tvm/runtime/memory.h>

namespace oneflow {
namespace xrt {
namespace of_tvm {

tvm::Array<tvm::relay::IndexExpr> Calc2DPadding(const std::string& padding_format,
                                                const std::vector<int32_t>& input_size,
                                                const std::vector<int32_t>& filter_size,
                                                const std::vector<int32_t>& stride,
                                                const std::vector<int32_t>& dilation);

tvm::Array<tvm::relay::IndexExpr> Calc2DPadding4Pool(const std::string& data_format,
                                                     const std::string& padding_format,
                                                     const Shape& in_shape,
                                                     const std::vector<int32_t>& pool_size,
                                                     const std::vector<int32_t>& stride);

}  // namespace of_tvm
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_TVM_OPS_NN_UTIL_H_
