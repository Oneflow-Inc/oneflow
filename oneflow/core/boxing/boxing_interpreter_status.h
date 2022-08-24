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
#ifndef ONEFLOW_CORE_BOXING_BOXING_INTERPRETER_STATUS_H_
#define ONEFLOW_CORE_BOXING_BOXING_INTERPRETER_STATUS_H_

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/framework/placed_nd_sbp.h"
#include "oneflow/core/common/shape.h"

namespace oneflow {

class BoxingInterpreterStatus;

extern Maybe<BoxingInterpreterStatus> (*MakeBoxingInterpreterStatus)(const std::string& boxing_name,
                                                                     const Shape& logical_shape,
                                                                     Symbol<PlacedNdSbp> in,
                                                                     Symbol<PlacedNdSbp> out);

extern Maybe<BoxingInterpreterStatus> (*MakeComposedBoxingInterpreterStatus)(
    const std::shared_ptr<BoxingInterpreterStatus>& lhs_status,
    const std::shared_ptr<BoxingInterpreterStatus>& rhs_status);

class BoxingInterpreterStatus final {
 public:
  BoxingInterpreterStatus(Symbol<std::vector<std::string>> sorted_boxing_names,
                          const Shape& logical_shape, Symbol<PlacedNdSbp> src_placed_nd_sbp,
                          Symbol<std::vector<Symbol<PlacedNdSbp>>> mid_placed_nd_sbp,
                          Symbol<PlacedNdSbp> dst_placed_nd_sbp)
      : sorted_boxing_names_(sorted_boxing_names),
        logical_shape_(logical_shape),
        src_placed_nd_sbp_(src_placed_nd_sbp),
        mid_placed_nd_sbp_(mid_placed_nd_sbp),
        dst_placed_nd_sbp_(dst_placed_nd_sbp) {}
  BoxingInterpreterStatus(Symbol<std::vector<std::string>> sorted_boxing_names,
                          const Shape& logical_shape, Symbol<PlacedNdSbp> src_placed_nd_sbp,
                          Symbol<PlacedNdSbp> dst_placed_nd_sbp)
      : BoxingInterpreterStatus(sorted_boxing_names, logical_shape, src_placed_nd_sbp,
                                SymbolOf(std::vector<Symbol<PlacedNdSbp>>()), dst_placed_nd_sbp) {}
  ~BoxingInterpreterStatus() = default;

  bool operator==(const BoxingInterpreterStatus& other) const {
    return this->sorted_boxing_names_ == other.sorted_boxing_names_
           && this->src_placed_nd_sbp_ == other.src_placed_nd_sbp_
           && this->mid_placed_nd_sbp_ == other.mid_placed_nd_sbp_
           && this->dst_placed_nd_sbp_ == other.dst_placed_nd_sbp_;
  }

  // Getters
  Symbol<std::vector<std::string>> sorted_boxing_names() const { return sorted_boxing_names_; }
  const Shape& logical_shape() const { return logical_shape_; }
  Symbol<PlacedNdSbp> src_placed_nd_sbp() const { return src_placed_nd_sbp_; }
  Symbol<PlacedNdSbp> dst_placed_nd_sbp() const { return dst_placed_nd_sbp_; }
  Symbol<std::vector<Symbol<PlacedNdSbp>>> mid_placed_nd_sbp() const { return mid_placed_nd_sbp_; }

  const std::string& boxing_routing() const;
  const std::string& nd_sbp_routing() const;
  const std::string& placement_routing() const;

 private:
  Symbol<std::vector<std::string>> sorted_boxing_names_;
  const Shape logical_shape_;
  Symbol<PlacedNdSbp> src_placed_nd_sbp_;
  Symbol<std::vector<Symbol<PlacedNdSbp>>> mid_placed_nd_sbp_;
  Symbol<PlacedNdSbp> dst_placed_nd_sbp_;
};

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::BoxingInterpreterStatus> {
  size_t operator()(const oneflow::BoxingInterpreterStatus& status) const {
    using namespace oneflow;
    size_t ret = 0;
    for (const auto& boxing_name : *status.sorted_boxing_names()) { AddHash(&ret, boxing_name); }
    AddHash(&ret, *status.src_placed_nd_sbp());
    for (const auto& mid_placed_nd_sbp : *status.mid_placed_nd_sbp()) {
      AddHash(&ret, *mid_placed_nd_sbp);
    }
    AddHash(&ret, *status.dst_placed_nd_sbp());
    return ret;
  }
};

}  // namespace std

#endif  // ONEFLOW_CORE_BOXING_BOXING_INTERPRETER_STATUS_H_
