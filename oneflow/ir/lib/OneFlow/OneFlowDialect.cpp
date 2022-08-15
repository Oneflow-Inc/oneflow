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
#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/OneFlowOps.h"
#include "OneFlow/OneFlowTypes.h"
#include "OneFlow/OneFlowOpsDialect.cpp.inc"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/TypeRange.h"

namespace mlir {

namespace oneflow {

void OneFlowDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "OneFlow/OneFlowOps.cpp.inc"
      ,
#define GET_OP_LIST
#include "OneFlow/OneFlow.assign_ops.cpp.inc"
      ,
#define GET_OP_LIST
#include "OneFlow/OneFlow.binary_ops.cpp.inc"
      ,
#define GET_OP_LIST
#include "OneFlow/OneFlow.broadcast_ops.cpp.inc"
      ,
#define GET_OP_LIST
#include "OneFlow/OneFlow.conv_ops.cpp.inc"
      ,
#define GET_OP_LIST
#include "OneFlow/OneFlow.cross_entropy_ops.cpp.inc"
      ,
#define GET_OP_LIST
#include "OneFlow/OneFlow.cuda_ops.cpp.inc"
      ,
#define GET_OP_LIST
#include "OneFlow/OneFlow.dataset_ops.cpp.inc"
      ,
#define GET_OP_LIST
#include "OneFlow/OneFlow.detection_ops.cpp.inc"
      ,
#define GET_OP_LIST
#include "OneFlow/OneFlow.eager_ops.cpp.inc"
      ,
#define GET_OP_LIST
#include "OneFlow/OneFlow.fused_ops.cpp.inc"
      ,
#define GET_OP_LIST
#include "OneFlow/OneFlow.idempotent_ops.cpp.inc"
      ,
#define GET_OP_LIST
#include "OneFlow/OneFlow.identity_ops.cpp.inc"
      ,
#define GET_OP_LIST
#include "OneFlow/OneFlow.image_ops.cpp.inc"
      ,
#define GET_OP_LIST
#include "OneFlow/OneFlow.indices_ops.cpp.inc"
      ,
#define GET_OP_LIST
#include "OneFlow/OneFlow.involution_ops.cpp.inc"
      ,
#define GET_OP_LIST
#include "OneFlow/OneFlow.loss_ops.cpp.inc"
      ,
#define GET_OP_LIST
#include "OneFlow/OneFlow.math_ops.cpp.inc"
      ,
#define GET_OP_LIST
#include "OneFlow/OneFlow.matmul_ops.cpp.inc"
      ,
#define GET_OP_LIST
#include "OneFlow/OneFlow.misc_ops.cpp.inc"
      ,
#define GET_OP_LIST
#include "OneFlow/OneFlow.nccl_ops.cpp.inc"
      ,
#define GET_OP_LIST
#include "OneFlow/OneFlow.normalization_ops.cpp.inc"
      ,
#define GET_OP_LIST
#include "OneFlow/OneFlow.optimizer_ops.cpp.inc"
      ,
#define GET_OP_LIST
#include "OneFlow/OneFlow.padding_ops.cpp.inc"
      ,
#define GET_OP_LIST
#include "OneFlow/OneFlow.parallel_cast_ops.cpp.inc"
      ,
#define GET_OP_LIST
#include "OneFlow/OneFlow.pool_ops.cpp.inc"
      ,
#define GET_OP_LIST
#include "OneFlow/OneFlow.quantization_ops.cpp.inc"
      ,
#define GET_OP_LIST
#include "OneFlow/OneFlow.reduce_ops.cpp.inc"
      ,
#define GET_OP_LIST
#include "OneFlow/OneFlow.reshape_ops.cpp.inc"
      ,
#define GET_OP_LIST
#include "OneFlow/OneFlow.scalar_ops.cpp.inc"
      ,
#define GET_OP_LIST
#include "OneFlow/OneFlow.softmax_ops.cpp.inc"
      ,
#define GET_OP_LIST
#include "OneFlow/OneFlow.summary_ops.cpp.inc"
      ,
#define GET_OP_LIST
#include "OneFlow/OneFlow.tensor_buffer_ops.cpp.inc"
      ,
#define GET_OP_LIST
#include "OneFlow/OneFlow.trigonometric_ops.cpp.inc"
      ,
#define GET_OP_LIST
#include "OneFlow/OneFlow.unary_ops.cpp.inc"
      ,
#define GET_OP_LIST
#include "OneFlow/OneFlow.upsample_ops.cpp.inc"
      ,
#define GET_OP_LIST
#include "OneFlow/OneFlow.one_embedding_ops.cpp.inc"
      ,
#define GET_OP_LIST
#include "OneFlow/OneFlow.linear_algebra_ops.cpp.inc"
      ,
#define GET_OP_LIST
#include "OneFlow/OneFlow.system_ops.cpp.inc"
      ,
#define GET_OP_LIST
#include "OneFlow/OneFlow.mlir_jit_ops.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "OneFlow/OneFlowOpsTypes.cpp.inc"
      >();
}

mlir::Operation* OneFlowDialect::materializeConstant(mlir::OpBuilder& builder,
                                                     mlir::Attribute value, mlir::Type type,
                                                     mlir::Location loc) {
  return builder.create<FrozenVariableOp>(loc, type, ValueRange(),
                                          value.cast<mlir::DictionaryAttr>().getValue());
}

}  // namespace oneflow

}  // namespace mlir
