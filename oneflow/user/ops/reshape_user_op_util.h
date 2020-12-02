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
#ifndef ONEFLOW_USER_OPS_RESHAPE_USER_OP_UTIL
#define ONEFLOW_USER_OPS_RESHAPE_USER_OP_UTIL

#include "oneflow/core/framework/sbp_context.h"
#include "oneflow/core/framework/user_op_conf.h"

namespace oneflow {
struct ReshapeUserOpUtil {
  static Maybe<Shape> GetLogicalOutBlobShape(const Shape& in_shape,
                                             const ShapeProto& reshape_proto);
  static Maybe<void> Squeeze(const Shape& origin, Shape* shape,
                             HashMap<int, int>* squeezed_axis2origin_axis);
  static Maybe<void> GetGroupStartInAxis2OutAxis(const Shape& in_shape, const Shape& out_shape,
                                                 const int64_t parallel_num,
                                                 HashMap<int, int>* group_start_in_axis2out_axis);
  static Maybe<void> GetReshapeUserOpSbpSignatures(const Shape& in_shape, const Shape& out_shape,
                                                   std::vector<user_op::OpArg> in_args,
                                                   std::vector<user_op::OpArg> out_args,
                                                   user_op::SbpContext* ctx);
};
}  // namespace oneflow

#endif  // ONEFLOW_USER_OPS_RESHAPE_USER_OP_UTIL
