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
#include "OneFlow/OneFlowOps.h"

namespace mlir {

namespace oneflow {

template<typename OpTy>
bool isLinearMatmulOp(OpTy op) {
  const bool isAlphaOne = op.alpha().convertToDouble() == 1.0;
  const bool isLinear = op.transpose_a() == false && op.transpose_b() == true;
  const bool hasNoAddToOutput = !op._add_to_output();
  const bool isCUDA = op.device_tag() == "cuda";
  return isAlphaOne && isLinear && hasNoAddToOutput && isCUDA;
}

bool MatmulOp::isLinear() { return isLinearMatmulOp(*this); }

bool BroadcastMatmulOp::isLinear() { return isLinearMatmulOp(*this); }

bool BiasAddOp::isLastDim() {
  return axis() == -1 || axis() == out().getType().cast<ShapedType>().getRank() - 1;
}

Value BroadcastAddOp::b() { return y(); }

Value BroadcastAddOp::out() { return z(); }

bool BroadcastAddOp::isLastDim() { return true; }

}  // namespace oneflow

}  // namespace mlir
