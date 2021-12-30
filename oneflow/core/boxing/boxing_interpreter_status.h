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

namespace oneflow {

class BoxingInterpreterStatus;
class PlacedNdSbp;

Maybe<BoxingInterpreterStatus> MakeBoxingInterpreterStatus(const std::string& boxing_name,
                                                           Symbol<PlacedNdSbp> in,
                                                           Symbol<PlacedNdSbp> out);
Maybe<BoxingInterpreterStatus> MakeComposedBoxingInterpreterStatus(
    const BoxingInterpreterStatus& lhs_status, const BoxingInterpreterStatus& rhs_status);

class BoxingInterpreterStatus final {
 public:
  BoxingInterpreterStatus(const std::string& boxing_name) : boxing_name_(boxing_name) {}
  BoxingInterpreterStatus(const std::string& boxing_name, Symbol<PlacedNdSbp> src_placed_nd_sbp,
                          Symbol<std::vector<Symbol<PlacedNdSbp>>> mid_placed_nd_sbp,
                          Symbol<PlacedNdSbp> dst_placed_nd_sbp)
      : boxing_name_(boxing_name),
        src_placed_nd_sbp_(src_placed_nd_sbp),
        mid_placed_nd_sbp_(mid_placed_nd_sbp),
        dst_placed_nd_sbp_(dst_placed_nd_sbp) {}
  BoxingInterpreterStatus(const std::string& boxing_name, Symbol<PlacedNdSbp> src_placed_nd_sbp,
                          Symbol<PlacedNdSbp> dst_placed_nd_sbp)
      : BoxingInterpreterStatus(boxing_name, src_placed_nd_sbp,
                                SymbolOf(std::vector<Symbol<PlacedNdSbp>>()), dst_placed_nd_sbp) {}
  ~BoxingInterpreterStatus() = default;

  // Getters
  const std::string& boxing_name() const { return boxing_name_; }
  Symbol<PlacedNdSbp> src_placed_nd_sbp() const { return src_placed_nd_sbp_; }
  Symbol<PlacedNdSbp> dst_placed_nd_sbp() const { return dst_placed_nd_sbp_; }
  Symbol<std::vector<Symbol<PlacedNdSbp>>> mid_placed_nd_sbp() const { return mid_placed_nd_sbp_; }
  const std::string& nd_sbp_routing() const;
  const std::string& placement_routing() const;

 private:
  std::string boxing_name_;
  Symbol<PlacedNdSbp> src_placed_nd_sbp_;
  Symbol<std::vector<Symbol<PlacedNdSbp>>> mid_placed_nd_sbp_;
  Symbol<PlacedNdSbp> dst_placed_nd_sbp_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_BOXING_BOXING_INTERPRETER_STATUS_H_
