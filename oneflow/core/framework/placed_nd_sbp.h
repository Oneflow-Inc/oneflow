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
#ifndef ONEFLOW_CORE_FRAMEWORK_PLACED_ND_SBP_H_
#define ONEFLOW_CORE_FRAMEWORK_PLACED_ND_SBP_H_

#include <functional>
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

class NdSbp;
class ParallelDesc;

class PlacedNdSbp final {
 public:
  PlacedNdSbp(const Symbol<NdSbp>& nd_sbp, const Symbol<ParallelDesc>& placement)
      : nd_sbp_(nd_sbp), placement_(placement) {}
  ~PlacedNdSbp() = default;

  static Maybe<Symbol<PlacedNdSbp>> (*New)(const Symbol<NdSbp>&, const Symbol<ParallelDesc>&);

  const Symbol<NdSbp>& nd_sbp() const { return nd_sbp_; }
  const Symbol<ParallelDesc>& placement() const { return placement_; }

  bool operator==(const PlacedNdSbp& other) const {
    return this->nd_sbp_ == other.nd_sbp_ && this->placement_ == other.placement_;
  }

 private:
  Symbol<NdSbp> nd_sbp_;
  Symbol<ParallelDesc> placement_;
};

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::PlacedNdSbp> final {
  size_t operator()(const oneflow::PlacedNdSbp& placed_nd_sbp) const {
    return oneflow::Hash(placed_nd_sbp.nd_sbp(), placed_nd_sbp.placement());
  }
};

}  // namespace std

#endif  // ONEFLOW_CORE_FRAMEWORK_PLACED_ND_SBP_H_
