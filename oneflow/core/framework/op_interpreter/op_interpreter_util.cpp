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
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include <cstddef>
#include <memory>

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/eager/foreign_boxing_util.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/framework/py_distribute.h"
#include "oneflow/core/framework/tensor_impl.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {
namespace one {

namespace {

std::shared_ptr<AutogradInterpreter> BuildEagerInterpreter(const bool& is_mirrored) {
  std::shared_ptr<OpExprInterpreter> internal;
  if (is_mirrored) {
    internal = std::make_shared<EagerMirroredInterpreter>();
  } else {
    internal = std::make_shared<EagerConsistentInterpreter>();
  }
  return std::make_shared<AutogradInterpreter>(internal);
}

std::shared_ptr<AutogradInterpreter> BuildLazyInterpreter() {
  auto internal = std::make_shared<LazyInterpreter>();
  return std::make_shared<AutogradInterpreter>(internal);
}

std::string ErrorString4Inputs(const TensorTuple& inputs, const OpExpr& op_expr) {
  std::stringstream error_str;
  error_str << "Got input tensors with inconsistent attributes!\n"
            << "op_type_name: " << op_expr.op_type_name() << "\n"
            << "attributes of inputs is:\n";
  int32_t idx = 0;
  for (const auto& tensor : inputs) {
    if (tensor->is_local()) {
      error_str << "local";
    } else {
      error_str << "consistent";
    }
    if (++idx != inputs.size()) { error_str << ", "; }
  }
  return error_str.str();
}

Maybe<AutogradInterpreter> GetInterpreter(const TensorTuple& inputs, const OpExprInterpContext& ctx,
                                          const OpExpr& op_expr) {
  static const auto& g_lazy_interpreter = BuildLazyInterpreter();
  static const auto& g_eager_consistent_interpreter = BuildEagerInterpreter(/*is_mirrored=*/false);
  static const auto& g_eager_mirrored_interpreter = BuildEagerInterpreter(/*is_mirrored=*/true);
  if (!LazyMode::is_enabled()) {
    if (inputs.empty()) {
      if (ctx.parallel_desc.has_value()) {
        JUST(ctx.nd_sbp);
        CHECK_OR_RETURN(!ctx.device.has_value());
        return g_eager_consistent_interpreter;
      } else {
        CHECK_OR_RETURN(!ctx.nd_sbp.has_value());
        return g_eager_mirrored_interpreter;
      }
    } else {
      if (inputs.at(0)->is_consistent()) {
        if (inputs.size() == 1) {
          // do nothing
        } else if (inputs.size() == 2) {
          CHECK_OR_RETURN(inputs.at(1)->is_consistent())
              << ErrorString4Inputs(inputs, op_expr);  // unroll loop for efficiency
        } else if (inputs.size() == 3) {
          CHECK_OR_RETURN(inputs.at(1)->is_consistent())
              << ErrorString4Inputs(inputs, op_expr);  // unroll loop for efficiency
          CHECK_OR_RETURN(inputs.at(2)->is_consistent())
              << ErrorString4Inputs(inputs, op_expr);  // unroll loop for efficiency
        } else {
          for (const auto& tensor : inputs) {
            CHECK_OR_RETURN(tensor->is_consistent()) << ErrorString4Inputs(inputs, op_expr);
          }
        }
        return g_eager_consistent_interpreter;
      } else {
        if (inputs.size() == 1) {
          // do nothing
        } else if (inputs.size() == 2) {
          CHECK_OR_RETURN(inputs.at(1)->is_local())
              << ErrorString4Inputs(inputs, op_expr);  // unroll loop for efficiency
        } else if (inputs.size() == 3) {
          CHECK_OR_RETURN(inputs.at(1)->is_local())
              << ErrorString4Inputs(inputs, op_expr);  // unroll loop for efficiency
          CHECK_OR_RETURN(inputs.at(2)->is_local())
              << ErrorString4Inputs(inputs, op_expr);  // unroll loop for efficiency
        } else {
          for (const auto& tensor : inputs) {
            CHECK_OR_RETURN(tensor->is_local()) << ErrorString4Inputs(inputs, op_expr);
          }
        }
        return g_eager_mirrored_interpreter;
      }
    }
    UNIMPLEMENTED_THEN_RETURN();
  }
  return g_lazy_interpreter;
}

}  // namespace

template<>
/* static */ Maybe<TensorTuple> OpInterpUtil::Dispatch<TensorTuple>(
    const OpExpr& op_expr, const TensorTuple& inputs, const OpExprInterpContext& ctx) {
  auto outputs = std::make_shared<TensorTuple>(op_expr.output_size());
  JUST(Dispatch(op_expr, inputs, outputs.get(), ctx));
  return outputs;
}

template<>
/* static */ Maybe<Tensor> OpInterpUtil::Dispatch<Tensor>(const OpExpr& op_expr,
                                                          const TensorTuple& inputs,
                                                          const OpExprInterpContext& ctx) {
  return JUST(Dispatch<TensorTuple>(op_expr, inputs, ctx))->at(0);
}

/* static */ Maybe<void> OpInterpUtil::Dispatch(const OpExpr& op_expr, const TensorTuple& inputs,
                                                TensorTuple* outputs,
                                                const OpExprInterpContext& ctx) {
  return JUST(GetInterpreter(inputs, ctx, op_expr))->Apply(op_expr, inputs, outputs, ctx);
}

/* static */ Maybe<OpAttribute> OpInterpUtil::AddOpAndInferOpAttribute(
    const OperatorConf& op_conf, const bool is_mirrored_strategy_enabled) {
  std::shared_ptr<OpAttribute> op_attribute = JUST([&]() -> Maybe<OpAttribute> {
    auto infer_ctx = JUST(GetCurInferCtx());
    if (is_mirrored_strategy_enabled) {
      return infer_ctx->AddAndInferMirroredOp(op_conf);
    } else {
      return infer_ctx->AddAndInferConsistentOp(op_conf);
    }
  }());
  return op_attribute;
}

/* static */ Maybe<OperatorConf> OpInterpUtil::GenBuiltinOpConf(const BuiltinOpExpr& op_expr,
                                                                const AttrMap& attrs) {
  auto op_conf = std::make_shared<OperatorConf>();
  JUST(op_expr.BuildOpConf(op_conf.get(), attrs));
  return op_conf;
}

/* static */ Maybe<Tensor> OpInterpUtil::BuildTensor(
    const std::shared_ptr<compatible_py::OpArgBlobAttribute>& blob_attr,
    const std::shared_ptr<compatible_py::OpArgParallelAttribute>& parallel_attr, const bool is_lazy,
    const bool is_local) {
  const auto& dtype = DataType(blob_attr->get_dtype());
  if (is_local) {
    const auto& device =
        JUST(Device::MakeDeviceByParallelDesc(*parallel_attr->parallel_desc_symbol()));
    const auto& tensor = JUST(MirroredTensor::MakeTensor(
        blob_attr->shape(), dtype, device, is_lazy, /*requires_grad=*/false, /*is_leaf=*/true));
    return static_cast<std::shared_ptr<Tensor>>(tensor);
  } else {
    const auto& nd_sbp = std::make_shared<NdSbp>();
    *nd_sbp->mutable_sbp_parallel()->Add() = *(parallel_attr->sbp_parallel());
    const auto& tensor =
        JUST(ConsistentTensor::MakeTensor(blob_attr->shape(), dtype, SymbolOf(*nd_sbp),
                                          SymbolOf(*parallel_attr->parallel_desc_symbol()), is_lazy,
                                          /*requires_grad=*/false, /*is_leaf=*/true));
    return static_cast<std::shared_ptr<Tensor>>(tensor);
  }
}

/* static */ Maybe<void> OpInterpUtil::CheckTensorMatchAttr(
    const std::shared_ptr<Tensor>& tensor,
    const std::shared_ptr<compatible_py::OpArgBlobAttribute>& blob_attr,
    const std::shared_ptr<compatible_py::OpArgParallelAttribute>& parallel_attr, const bool is_lazy,
    const bool is_local, const bool requires_grad, const bool is_leaf) {
  CHECK_EQ_OR_RETURN(*tensor->shape(), *blob_attr->shape());
  CHECK_EQ_OR_RETURN(tensor->is_lazy(), is_lazy);
  CHECK_EQ_OR_RETURN(tensor->is_local(), is_local);
  const auto& dtype = DataType(blob_attr->get_dtype());
  CHECK_EQ_OR_RETURN(tensor->dtype()->data_type(), dtype);
  CHECK_EQ_OR_RETURN(tensor->requires_grad(), requires_grad);
  CHECK_EQ_OR_RETURN(tensor->is_leaf(), is_leaf);
  if (is_local) {
    const auto& device =
        JUST(Device::MakeDeviceByParallelDesc(*parallel_attr->parallel_desc_symbol()));
    CHECK_OR_RETURN(JUST(tensor->device()) == device);
  } else {
    const auto& nd_sbp = std::make_shared<NdSbp>();
    *nd_sbp->mutable_sbp_parallel()->Add() = *(parallel_attr->sbp_parallel());
    CHECK_OR_RETURN(JUST(tensor->nd_sbp()) == SymbolOf(*nd_sbp));
    CHECK_OR_RETURN(JUST(tensor->parallel_desc())
                    == SymbolOf(*parallel_attr->parallel_desc_symbol()));
  }
  return Maybe<void>::Ok();
}

}  // namespace one
}  // namespace oneflow
