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
#ifndef ONEFLOW_IR_INCLUDE_SBP_SBPIMPORTER_H_
#define ONEFLOW_IR_INCLUDE_SBP_SBPIMPORTER_H_
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/sbp_parallel.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "OneFlow/OneFlowOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"

#include <functional>
#include <string>

namespace mlir {
namespace oneflow {

class SBPTranslation {
 public:
  static mlir::LogicalResult PrintSbpAttrToString(mlir::Attribute sbp_attr, std::string& sbp);
  static mlir::Attribute ConvertSBPToString(mlir::Builder& builder,
                                            mlir::sbp::ParallelSignatureAttr& parallel);
  static mlir::Attribute ConvertNdSbpToPsig(mlir::Builder& builder,
                                            const std::vector<std::string>& nd_sbp,
                                            const int nd_size);
};

}  // namespace oneflow
}  // namespace mlir

#endif  // ONEFLOW_IR_INCLUDE_SBP_SBPIMPORTER_H_
