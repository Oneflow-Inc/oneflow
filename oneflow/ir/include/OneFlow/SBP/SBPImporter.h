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

mlir::LogicalResult PrintSbpAttrToString(mlir::Attribute sbp_attr, std::string* sbp);
mlir::Attribute ConvertSBPToString(mlir::Builder& builder,
                                   mlir::sbp::ParallelSignatureAttr& parallel);
mlir::Attribute ConvertNdSbpToPsig(mlir::Builder& builder, const std::vector<std::string>& nd_sbp,
                                   const int nd_size);
}  // namespace oneflow
}  // namespace mlir

#endif  // ONEFLOW_IR_INCLUDE_SBP_SBPIMPORTER_H_
