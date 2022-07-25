#include "OneFlow/SBP/SBPImporter.h"

#include <vector>
#include <string>

namespace mlir {
namespace oneflow {

mlir::LogicalResult PrintSbpAttrToString(mlir::Attribute sbp_attr, std::string* sbp) {
  if (auto sbp_s_attr = sbp_attr.dyn_cast<mlir::sbp::SplitAttr>()) {
    *sbp = "S(" + std::to_string(sbp_s_attr.getAxis()) + ")";
  } else if (auto sbp_b_attr = sbp_attr.dyn_cast<mlir::sbp::BroadcastAttr>()) {
    *sbp = "B";
  } else if (auto sbp_p_attr = sbp_attr.dyn_cast<mlir::sbp::PartialSumAttr>()) {
    *sbp = "P";
  } else {
    return mlir::failure();
  }
  return mlir::success();
}
mlir::Attribute ConvertSBPToString(mlir::Builder& builder,
                                   mlir::sbp::ParallelSignatureAttr& parallel) {
  std::vector<std::string> list;
  for (auto output : parallel.getOutputs()) {
    if (auto nd_outputs = output.dyn_cast<mlir::ArrayAttr>()) {
      for (auto nd_output : nd_outputs) {
        std::string sbp;
        if (failed(PrintSbpAttrToString(nd_output, &sbp))) return {};
        list.push_back(sbp);
      }
    } else {
      std::string sbp;
      if (failed(PrintSbpAttrToString(output, &sbp))) return {};
      list.push_back(sbp);
    }
  }
  return builder.getStrArrayAttr(
      makeArrayRef(llvm::SmallVector<llvm::StringRef>(list.begin(), list.end())));
}

mlir::Attribute ConvertNdSbpToPsig(mlir::Builder& builder, std::vector<std::string>& nd_sbp,
                                   const int nd_size) {
  auto ctx = builder.getContext();
  std::vector<mlir::Attribute> outputs_vec;
  for (const auto& sbp_data : nd_sbp) {
    ::oneflow::SbpParallel sbp;
    ParseSbpParallelFromString(sbp_data, &sbp);
    mlir::Attribute attr;
    if (sbp.has_split_parallel()) {
      attr = mlir::sbp::SplitAttr::get(ctx, sbp.split_parallel().axis());
    } else if (sbp.has_broadcast_parallel()) {
      attr = mlir::sbp::BroadcastAttr::get(ctx);
    } else if (sbp.has_partial_sum_parallel()) {
      attr = mlir::sbp::PartialSumAttr::get(ctx);
    } else {
      llvm::errs() << "Unsupported sbp type from nd_sbp: ";
      for (const auto& sbp_data : nd_sbp) { llvm::errs() << sbp_data << " "; }
      llvm::errs() << "\n";
      exit(EXIT_FAILURE);
    }
    outputs_vec.push_back(attr);
  }

  auto inputs = builder.getArrayAttr({});
  mlir::ArrayAttr outputs;

  std::vector<mlir::Attribute> outputs_vec_nd;
  for (auto iter = outputs_vec.begin(); iter < outputs_vec.end(); iter += nd_size) {
    outputs_vec_nd.emplace_back(
        builder.getArrayAttr(std::vector<mlir::Attribute>(iter, iter + nd_size)));
  }
  outputs = builder.getArrayAttr(outputs_vec_nd);
  auto res = mlir::sbp::ParallelSignatureAttr::get(ctx, inputs, outputs);
  return res;
}
}  // namespace oneflow
}  // namespace mlir
