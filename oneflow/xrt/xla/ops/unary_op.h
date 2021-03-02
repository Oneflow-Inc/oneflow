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
#ifndef ONEFLOW_XRT_XLA_OPS_UNARY_OP_H_
#define ONEFLOW_XRT_XLA_OPS_UNARY_OP_H_

#include "oneflow/xrt/xla/xla_data_type.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace oneflow {
namespace xrt {
namespace mola {
namespace op {

#define OFXLA_DECLARE_UNARY_OP(op)                                    \
  struct op {                                                         \
    xla::XlaOp operator()(const xla::XlaOp &x) { return xla::op(x); } \
  };

OFXLA_DECLARE_UNARY_OP(Abs);
OFXLA_DECLARE_UNARY_OP(Logistic);
OFXLA_DECLARE_UNARY_OP(Tanh);
OFXLA_DECLARE_UNARY_OP(Rsqrt);

#undef OFXLA_DECLARE_UNARY_OP

}  // namespace op
}  // namespace mola
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_XLA_OPS_UNARY_OP_H_
