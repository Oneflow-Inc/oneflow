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

Value MatmulOp::matMulGetX() { return a(); }

Value MatmulOp::matMulGetW() { return b(); }

Value MatmulOp::matMulGetY() { return out(); }

bool BroadcastMatmulOp::isLinear() { return isLinearMatmulOp(*this); }

Value BroadcastMatmulOp::matMulGetX() { return a(); }

Value BroadcastMatmulOp::matMulGetW() { return b(); }

Value BroadcastMatmulOp::matMulGetY() { return out(); }

bool BiasAddOp::isLastDim() {
  return axis() == -1 || axis() == out().getType().cast<ShapedType>().getRank() - 1;
}

Value BiasAddOp::biasAddGetBias() { return b(); }

Value BiasAddOp::biasAddGetOut() { return out(); }

Value BroadcastAddOp::biasAddGetBias() { return y(); }

Value BroadcastAddOp::biasAddGetOut() { return z(); }

bool BroadcastAddOp::isLastDim() { return true; }

Value FusedMatmulBiasOp::matMulGetX() { return x(); }

Value FusedMatmulBiasOp::matMulGetW() { return weight(); }

Value FusedMatmulBiasOp::matMulGetY() { return out(); }

namespace {

bool shouldGroupFusedMatmulBiasOp(FusedMatmulBiasOp& op) {
  return !op._add_to_output() && op.device_tag() == "cuda" && op.alpha().convertToDouble() == 1.0;
}

}  // namespace

bool FusedMatmulBiasOp::isLinear() { return shouldGroupFusedMatmulBiasOp(*this); }

bool FusedMatmulBiasOp::isLastDim() { return shouldGroupFusedMatmulBiasOp(*this); }

Value FusedMatmulBiasOp::biasAddGetBias() { return bias(); }

Value FusedMatmulBiasOp::biasAddGetOut() { return out(); }

}  // namespace oneflow

}  // namespace mlir
