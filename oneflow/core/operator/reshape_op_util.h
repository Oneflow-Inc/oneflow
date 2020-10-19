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
#ifndef ONEFLOW_CORE_OPERATOR_RESHAPE_OP_UTIL_H_
#define ONEFLOW_CORE_OPERATOR_RESHAPE_OP_UTIL_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {
struct ReshapeOpUtil {
  static Maybe<Shape> GetLogicalOutBlobShape(const Shape& in_shape,
                                             const ShapeProto& reshape_proto);
  static Maybe<void> Squeeze(const Shape& origin, Shape* shape,
                             HashMap<int, int>* squeezed_axis2origin_axis);
  static Maybe<void> GetGroupStartInAxis2OutAxis(const Shape& in_shape, const Shape& out_shape,
                                                 const int64_t parallel_num,
                                                 HashMap<int, int>* group_start_in_axis2out_axis);
  static Maybe<void> GetReshapeSbpSignatures(const Shape& in_shape, const Shape& out_shape,
                                             const PbRpf<std::string>& input_bns,
                                             const PbRpf<std::string>& output_bns,
                                             const int64_t parallel_num,
                                             SbpSignatureList* sbp_sig_list);
};
}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_RESHAPE_OP_UTIL_H_
