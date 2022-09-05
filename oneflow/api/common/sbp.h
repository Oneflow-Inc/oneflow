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

#ifndef ONEFLOW_API_COMMON_SBP_H_
#define ONEFLOW_API_COMMON_SBP_H_

#include "oneflow/core/job/sbp_parallel.pb.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/common/maybe.h"

namespace oneflow {

namespace api {

// NOTE: The api inferface will print the whole name of sbp.

inline Maybe<std::string> ApiSbpToString(Symbol<SbpParallel> sbp_sym) {
  std::string sbp_str = "oneflow.sbp.";
  if (sbp_sym->has_broadcast_parallel()) {
    sbp_str += "broadcast";
  } else if (sbp_sym->has_partial_sum_parallel()) {
    sbp_str += "partial_sum";
  } else if (sbp_sym->has_split_parallel()) {
    sbp_str += "split(dim=" + std::to_string(sbp_sym->split_parallel().axis()) + ")";
  } else {
    UNIMPLEMENTED_THEN_RETURN();
  }
  return sbp_str;
}

inline Maybe<std::string> ApiNdSbpToString(Symbol<NdSbp> nd_sbp) {
  std::string str = "(";
  for (int i = 0; i < nd_sbp->sbp_parallel_size(); ++i) {
    if (i > 0) { str += ", "; }
    str += *JUST(ApiSbpToString(SymbolOf(nd_sbp->sbp_parallel(i))));
  }
  if (nd_sbp->sbp_parallel_size() == 1) { str += ","; }
  str += ")";
  return str;
}

}  // namespace api

}  // namespace oneflow

#endif  // !ONEFLOW_API_COMMON_SBP_H_
