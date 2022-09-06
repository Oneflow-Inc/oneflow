#include "OneFlow/OneFlowOps.h"

namespace mlir {

namespace oneflow {

namespace user_op {

template<template<typename T> class Trait>
LogicalResult GetFilteredSegmentKeyAndSizes(Operation* op, std::vector<std::string>& keys,
                                            std::vector<int32_t>& sizes);

using ArgID = std::pair<std::string, int32_t>;

template<template<typename T> class Trait>
class ArgIds {
 public:
  explicit ArgIds(Operation* op);
  std::vector<ArgID>::const_iterator begin() const { return ids_.begin(); }
  std::vector<ArgID>::const_iterator end() const { return ids_.end(); }

 private:
  std::vector<ArgID> ids_;
};

}  // namespace user_op

}  // namespace oneflow

}  // namespace mlir
