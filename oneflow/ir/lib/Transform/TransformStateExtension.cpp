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
#include "Transform/TransformStateExtension.h"

using namespace mlir;

LogicalResult mlir::oneflow::transform_dialect::TransformStateExtension::updateMapping(
    Operation* previous, Operation* updated) {
  // Update value handles. The new ops should have at least as many results as
  // the replacement op. Fewer results are acceptable, if those results are not
  // mapped to any handle.
  for (auto r = updated->getNumResults(); r < previous->getNumResults(); ++r) {
    SmallVector<Value> handles;
    (void)getTransformState().getHandlesForPayloadValue(previous->getResult(r), handles);
    if (!handles.empty())
      return emitError(previous->getLoc())
             << "cannot replace an op with another op producing fewer results "
                "while tracking handles";
  }

  for (auto [oldValue, newValue] : llvm::zip(previous->getResults(), updated->getResults()))
    if (failed(replacePayloadValue(oldValue, newValue))) return failure();

  // Update op handle.
  return replacePayloadOp(previous, updated);
}
