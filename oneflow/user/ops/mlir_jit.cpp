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

#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/ops/nn_util.h"
#include "oneflow/core/framework/op_generated.h"
#include "oneflow/core/operator/operator.h"
#include <regex>

namespace oneflow {

namespace {

Maybe<void> InferTensorDesc(user_op::InferContext* ctx) {
  auto parse_mlir =
      [](const std::string& mlir) -> std::pair<std::pair<std::string, std::string>, std::string> {
    std::regex function_type_pattern("function_type = \\((.+?), (.+?)\\) -> (.+?),");
    std::smatch result;
    if (std::regex_search(mlir, result, function_type_pattern)) {
      return {{result.str(1), result.str(2)}, result.str(3)};
    }
    LOG(ERROR) << "Parse mlir assembly failed";
    return {};
  };

  auto parse_type = [](const std::string& type) -> DataType {
    std::string full_type;
    if (type.at(0) == 'i' || type.at(0) == 'u') {
      full_type = type.at(0) == 'i' ? "kInt" : "kUInt";
      full_type += type.substr(1);
      DataType val;
      if (DataType_Parse(full_type, &val)) { return val; }
    } else if (type.at(0) == 'f') {
      if ("f16" == type) return DataType::kFloat16;
      else if ("f32" == type) return DataType::kFloat;
      else if ("f64" == type) return DataType::kDouble;
    }
    LOG(ERROR) << "Parse type in mlir assembly failed";
    return {};
  };

  auto parse_tensor = [&](const std::string& tensor) -> std::pair<Shape, DataType> {
    std::regex tensor_type_pattern("tensor<(.*)x(.+?)>");
    std::regex num_pattern("(\\d+)");
    std::smatch result;
    DimVector res;
    if (regex_match(tensor, result, tensor_type_pattern)) {
      auto nums = result.str(1);
      for (std::sregex_iterator it(nums.begin(), nums.end(), num_pattern), end_it; it != end_it;
           ++it) {
        res.push_back(std::stoi(it->str(1)));
      }
      return {Shape(res), parse_type(result.str(2))};
    }
    LOG(ERROR) << "Parse tensor in mlir assembly failed";
    return {};
  };

  auto mlir_assembly = ctx->Attr<std::string>("mlir_assembly");
  auto tensors = parse_mlir(mlir_assembly);
  auto in0_info = parse_tensor(tensors.first.first);
  auto in1_info = parse_tensor(tensors.first.second);
  auto out0_info = parse_tensor(tensors.second);

  CHECK_EQ(in0_info.first, out0_info.first);
  CHECK_EQ(in0_info.first, ctx->InputShape("in", 0));

  CHECK_EQ(in1_info.second, out0_info.second);
  CHECK_EQ(in1_info.second, ctx->InputDType("in", 1));

  CHECK_EQ(ctx->inputs().size(), 2);
  CHECK_EQ(ctx->outputs().size(), 1);
  const Shape& in_shape = ctx->InputShape("in", 0);
  Shape* out_shape = ctx->MutOutputShape("out", 0);
  *out_shape = in_shape;
  *ctx->MutOutputDType("out", 0) = ctx->InputDType("in", 1);
  return Maybe<void>::Ok();
}

Maybe<void> GetSbpFn(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Broadcast(ctx->inputs()).Broadcast(ctx->outputs()).Build();
  return Maybe<void>::Ok();
}

Maybe<void> InferDataTypeFn(user_op::InferContext* ctx) {
  *ctx->MutOutputDType("out", 0) = ctx->InputDType("in", 0);
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<void> MlirJitOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return InferTensorDesc(ctx);
}

Maybe<void> MlirJitOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferTensorDesc(ctx);
}

Maybe<void> MlirJitOp::GetSbp(user_op::SbpContext* ctx) { return GetSbpFn(ctx); }

Maybe<void> MlirJitOp::InferDataType(user_op::InferContext* ctx) { return InferDataTypeFn(ctx); }

}  // namespace oneflow
