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
#include "oneflow/core/framework/framework.h"

namespace oneflow {
struct ReshapeUserOpUtil {
  static Maybe<Shape> GetLogicalOutBlobShape(const Shape& in_shape, const Shape& reshape);
  static Maybe<void> Squeeze(const Shape& origin, Shape* shape,
                             HashMap<int, int>* squeezed_axis2origin_axis);
  static Maybe<void> GetGroupStartInAxis2OutAxis(const Shape& in_shape, const Shape& out_shape,
                                                 const int64_t hierarchy_value,
                                                 HashMap<int, int>* group_start_in_axis2out_axis);
  static Maybe<void> GetReshapeUserOpSbpSignatures(const Shape& in_shape, const Shape& out_shape,
                                                   const std::vector<user_op::OpArg>& in_args,
                                                   const std::vector<user_op::OpArg>& out_args,
                                                   const int64_t hierarchy_value,
                                                   user_op::UserOpSbpSignatureBuilder* builder);

  static Maybe<void> DfsCombineNdSbpSignatureGroups(
      const std::vector<std::map<int, std::pair<int, int>>>& nd_sbp_sig_groups,
      size_t rank_num_axes, std::vector<std::vector<std::pair<int, int>>>* nd_sbp_sig_list);
  static Maybe<void> DfsCombineNdSbpSignatureGroups(
      const std::vector<std::map<int, std::pair<int, int>>>& nd_sbp_sig_groups,
      size_t rank_num_axes, const std::map<int, std::pair<int, int>>& nd_sbp_sig_group,
      std::set<std::vector<std::pair<int, int>>>& nd_sbp_sig_set);
  static Maybe<void> EnumerateNdSplitIn2OutAxis(
      const Shape& in_shape, const std::vector<int>& origin_in_axes, const Shape& out_shape,
      const std::vector<int>& origin_out_axes, const Shape& rank_mesh,
      std::vector<std::map<int, std::pair<int, int>>>* nd_split_groups);
  static Maybe<void> EnumerateNdSplitIn2OutAxisGroups(
      const Shape& in_shape, const Shape& out_shape, const Shape& rank_mesh,
      std::vector<std::map<int, std::pair<int, int>>>* nd_sbp_in2out_sig_groups);
  static Maybe<void> EnumerateNdSbpIn2OutSignatures(
      const Shape& in_shape, const Shape& out_shape, const Shape& rank_mesh,
      std::vector<std::vector<std::pair<int, int>>>* nd_sbp_in2out_signatures);
  static Maybe<void> FilterNdSbpIn2OutSignatures(
      const Shape& in_shape, const Shape& out_shape, const Shape& rank_mesh,
      std::vector<std::vector<std::pair<int, int>>>* nd_sbp_in2out_signatures);
  static Maybe<void> EnumerateNdSbpSignatures(const std::vector<user_op::OpArg>& in_args,
                                              const Shape& in_shape,
                                              const std::vector<user_op::OpArg>& out_args,
                                              const Shape& out_shape, const Shape& rank_mesh,
                                              std::vector<NdSbpSignature>* nd_sbp_sig_list);
};
}  // namespace oneflow

#endif  // ONEFLOW_USER_OPS_RESHAPE_USER_OP_UTIL
