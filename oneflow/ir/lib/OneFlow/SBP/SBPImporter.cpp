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
#include "OneFlow/SBP/SBPImporter.h"

#include <vector>
#include <string>

namespace mlir {
namespace oneflow {

mlir::LogicalResult SBPTranslation::PrintSbpAttrToString(mlir::Attribute sbp_attr,
                                                         std::string& sbp) {
  if (auto sbp_s_attr = sbp_attr.dyn_cast<mlir::sbp::SplitAttr>()) {
    sbp = "S(" + std::to_string(sbp_s_attr.getAxis()) + ")";
  } else if (auto sbp_b_attr = sbp_attr.dyn_cast<mlir::sbp::BroadcastAttr>()) {
    sbp = "B";
  } else if (auto sbp_p_attr = sbp_attr.dyn_cast<mlir::sbp::PartialSumAttr>()) {
    sbp = "P";
  } else if (auto sbp_p_attr = sbp_attr.dyn_cast<mlir::sbp::AnyAttr>()) {
    sbp = "";
  } else {
    return mlir::failure();
  }
  return mlir::success();
}
mlir::Attribute SBPTranslation::ConvertSBPToString(mlir::Builder& builder,
                                                   mlir::sbp::ParallelSignatureAttr& parallel) {
  std::vector<std::string> list;
  for (auto output : parallel.getOutputs()) {
    if (auto nd_outputs = output.dyn_cast<mlir::ArrayAttr>()) {
      for (auto nd_output : nd_outputs) {
        std::string sbp;
        if (failed(SBPTranslation::PrintSbpAttrToString(nd_output, sbp))) return {};
        list.push_back(sbp);
      }
    } else {
      std::string sbp;
      if (failed(SBPTranslation::PrintSbpAttrToString(output, sbp))) return {};
      list.push_back(sbp);
    }
  }
  return builder.getStrArrayAttr(
      makeArrayRef(llvm::SmallVector<llvm::StringRef>(list.begin(), list.end())));
}

mlir::Attribute SBPTranslation::ConvertNdSbpToPsig(mlir::Builder& builder,
                                                   const std::vector<std::string>& nd_sbp,
                                                   const int nd_size) {
  auto ctx = builder.getContext();
  std::vector<mlir::Attribute> outputs_vec;
  for (const auto& sbp_data : nd_sbp) {
    mlir::Attribute attr;
    if (sbp_data == "") {
      attr = mlir::sbp::AnyAttr::get(ctx);
    } else {
      ::oneflow::SbpParallel sbp;
      ParseSbpParallelFromString(sbp_data, &sbp);
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
  return mlir::sbp::ParallelSignatureAttr::get(ctx, inputs, outputs);
}
}  // namespace oneflow
}  // namespace mlir
