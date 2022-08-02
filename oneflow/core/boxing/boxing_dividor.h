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
#ifndef ONEFLOW_CORE_BOXING_BOXING_DIVIDOR_H_
#define ONEFLOW_CORE_BOXING_BOXING_DIVIDOR_H_

#include <functional>
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/symbol.h"

namespace oneflow {

class PlacedNdSbp;

class BoxingDividor final {
 public:
  BoxingDividor(const BoxingDividor&) = delete;
  BoxingDividor(BoxingDividor&&) = delete;
  ~BoxingDividor() = default;

  using FunctionT =
      std::function<Maybe<Symbol<PlacedNdSbp>>(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out)>;

  BoxingDividor(const std::string& name, const FunctionT& function)
      : name_(name), function_(function) {}

  const std::string& name() const { return name_; }

  Maybe<Symbol<PlacedNdSbp>> operator()(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out) const {
    return function_(in, out);
  }

 private:
  std::string name_;
  FunctionT function_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_BOXING_BOXING_DIVIDOR_H_
