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
#include <oneflow/ir/include/OneFlow/Transform/TransposeHelpers.h>

namespace mlir {

namespace oneflow {

RankedTensorType getNHWCType(RankedTensorType t) {
  return RankedTensorType::get({t.getShape()[0], t.getShape()[2], t.getShape()[3], t.getShape()[1]},
                               t.getElementType());
}

RankedTensorType getNHWCType(Type t) { return getNHWCType(t.cast<RankedTensorType>()); }
RankedTensorType getNHWCType(Value v) { return getNHWCType(v.getType()); }

RankedTensorType getNCHWType(RankedTensorType t) {
  return RankedTensorType::get({t.getShape()[0], t.getShape()[3], t.getShape()[1], t.getShape()[2]},
                               t.getElementType());
}
RankedTensorType getNCHWType(Type t) { return getNCHWType(t.cast<RankedTensorType>()); }
RankedTensorType getNCHWType(Value v) { return getNCHWType(v.getType()); }

llvm::SmallVector<Type, 4> getNHWCResultTypes(NCHWCompatible op) {
  llvm::SmallVector<Type, 4> result_types;
  llvm::DenseSet<Value> transpose_result = op.ResultsToTranspose();
  for (Value result : op->getOpResults()) {
    Type t = result.getType();
    if (transpose_result.find(result) != transpose_result.end()) {
      result_types.push_back(getNHWCType(t));
    } else {
      result_types.push_back(t);
    }
  }
  return result_types;
}

}  // namespace oneflow

}  // namespace mlir
