#ifndef ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_BOXING_BOXING_DIVIDOR_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_BOXING_BOXING_DIVIDOR_H_

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

}

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_BOXING_BOXING_DIVIDOR_H_
