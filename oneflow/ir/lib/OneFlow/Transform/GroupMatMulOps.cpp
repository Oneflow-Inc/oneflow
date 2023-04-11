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
  const bool isAlphaOne = op.getAlpha().convertToDouble() == 1.0;
  const bool isLinear = op.getTransposeA() == false && op.getTransposeB() == true;
  const bool hasNoAddToOutput = !op.get_addToOutput();
  const bool isCUDA = op.getDeviceTag() == "cuda";
  return isAlphaOne && isLinear && hasNoAddToOutput && isCUDA;
}

bool MatmulOp::isLinear() { return isLinearMatmulOp(*this); }

Value MatmulOp::matMulGetX() { return getA(); }

Value MatmulOp::matMulGetW() { return getB(); }

Value MatmulOp::matMulGetY() { return getOut(); }

bool BroadcastMatmulOp::isLinear() { return isLinearMatmulOp(*this); }

Value BroadcastMatmulOp::matMulGetX() { return getA(); }

Value BroadcastMatmulOp::matMulGetW() { return getB(); }

Value BroadcastMatmulOp::matMulGetY() { return getOut(); }

bool BiasAddOp::isLastDim() {
  return getAxis() == -1 || getAxis() == getOut().getType().cast<ShapedType>().getRank() - 1;
}

Value BiasAddOp::biasAddGetBias() { return getB(); }

Value BiasAddOp::biasAddGetOut() { return getOut(); }

Value BroadcastAddOp::biasAddGetBias() { return getY(); }

Value BroadcastAddOp::biasAddGetOut() { return getZ(); }

bool BroadcastAddOp::isLastDim() { return true; }

Value FusedMatmulBiasOp::matMulGetX() { return getX(); }

Value FusedMatmulBiasOp::matMulGetW() { return getWeight(); }

Value FusedMatmulBiasOp::matMulGetY() { return getOut(); }

namespace {

bool shouldGroupFusedMatmulBiasOp(FusedMatmulBiasOp& op) {
  return !op.get_addToOutput() && op.getDeviceTag() == "cuda"
         && op.getAlpha().convertToDouble() == 1.0;
}

}  // namespace

bool FusedMatmulBiasOp::isLinear() { return shouldGroupFusedMatmulBiasOp(*this); }

bool FusedMatmulBiasOp::isLastDim() { return shouldGroupFusedMatmulBiasOp(*this); }

Value FusedMatmulBiasOp::biasAddGetBias() { return getBias(); }

Value FusedMatmulBiasOp::biasAddGetOut() { return getOut(); }

}  // namespace oneflow

}  // namespace mlir
