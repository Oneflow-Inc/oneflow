#include "OneFlow/OneFlowOps.h"

namespace mlir {

namespace oneflow {

namespace user_op {

template<template<typename T> class Trait>
LogicalResult GetFilteredSegmentKeyAndSizes(Operation* op, std::vector<std::string>& keys,
                                            std::vector<int32_t>& sizes);

}  // namespace user_op

}  // namespace oneflow

}  // namespace mlir
