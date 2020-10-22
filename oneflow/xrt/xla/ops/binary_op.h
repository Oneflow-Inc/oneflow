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
#ifndef ONEFLOW_XRT_XLA_OPS_BINARY_OP_H_
#define ONEFLOW_XRT_XLA_OPS_BINARY_OP_H_

#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace oneflow {
namespace xrt {
namespace mola {
namespace op {

#define OFXLA_DECLARE_BINARY_OP(op)                                             \
  struct op {                                                                   \
    xla::XlaOp operator()(xla::XlaOp a, xla::XlaOp b) { return xla::op(a, b); } \
  };

OFXLA_DECLARE_BINARY_OP(Add);
OFXLA_DECLARE_BINARY_OP(Mul);
OFXLA_DECLARE_BINARY_OP(Div);
OFXLA_DECLARE_BINARY_OP(Min);

#undef OFXLA_DECLARE_BINARY_OP

}  // namespace op
}  // namespace mola
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_XLA_OPS_BINARY_OP_H_
