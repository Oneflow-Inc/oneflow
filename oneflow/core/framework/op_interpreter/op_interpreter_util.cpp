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
#include <memory>

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/framework/tensor_impl.h"
#include "oneflow/core/functional/tensor_processor.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/profiler/profiler.h"

namespace oneflow {
namespace one {

namespace {

std::shared_ptr<AutogradInterpreter> BuildEagerInterpreter(const bool& is_local) {
  std::shared_ptr<OpExprInterpreter> internal;
  if (is_local) {
    internal = std::make_shared<EagerLocalInterpreter>();
  } else {
    internal = std::make_shared<EagerGlobalInterpreter>();
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
      error_str << "global";
    }
    if (++idx != inputs.size()) { error_str << ", "; }
  }
  return error_str.str();
}

Maybe<AutogradInterpreter> GetInterpreter(const TensorTuple& inputs, const OpExprInterpContext& ctx,
                                          const OpExpr& op_expr) {
  static const auto& g_lazy_interpreter = BuildLazyInterpreter();
  static const auto& g_eager_global_interpreter = BuildEagerInterpreter(/*is_local=*/false);
  static const auto& g_eager_local_interpreter = BuildEagerInterpreter(/*is_local=*/true);
  bool is_local = true;
  if (inputs.empty()) {
    if (ctx.parallel_desc.has_value()) {
      JUST(ctx.nd_sbp);
      CHECK_OR_RETURN(!ctx.device.has_value());
      is_local = false;
    } else {
      CHECK_OR_RETURN(!ctx.nd_sbp.has_value());
    }
  } else {
    if (inputs[0]->is_global()) {
      if (inputs.size() == 1) {
        // do nothing
      } else if (inputs.size() == 2) {
        CHECK_OR_RETURN(inputs[1]->is_global())      // NOLINT
            << ErrorString4Inputs(inputs, op_expr);  // unroll loop for efficiency
      } else if (inputs.size() == 3) {
        CHECK_OR_RETURN(inputs[1]->is_global())
            << ErrorString4Inputs(inputs, op_expr);  // unroll loop for efficiency
        CHECK_OR_RETURN(inputs[2]->is_global())
            << ErrorString4Inputs(inputs, op_expr);  // unroll loop for efficiency
      } else {
        for (const auto& tensor : inputs) {
          CHECK_OR_RETURN(tensor->is_global()) << ErrorString4Inputs(inputs, op_expr);
        }
      }
      is_local = false;
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
    }
  }
  if (!LazyMode::is_enabled()) {
    if (is_local) {
      return g_eager_local_interpreter;
    } else {
      return g_eager_global_interpreter;
    }
  } else {
    return g_lazy_interpreter;
  }
}

}  // namespace

template<>
/* static */ Maybe<TensorTuple> OpInterpUtil::Dispatch<TensorTuple>(
    const OpExpr& op_expr, const TensorTuple& inputs, const OpExprInterpContext& ctx) {
  OF_PROFILER_RANGE_GUARD("Dispatch");
  auto outputs = std::make_shared<TensorTuple>(op_expr.output_size());
  JUST(Dispatch(op_expr, inputs, outputs.get(), ctx));
  return outputs;
}

template<>
/* static */ Maybe<Tensor> OpInterpUtil::Dispatch<Tensor>(const OpExpr& op_expr,
                                                          const TensorTuple& inputs,
                                                          const OpExprInterpContext& ctx) {
  OF_PROFILER_RANGE_GUARD("Dispatch");
  return JUST(Dispatch<TensorTuple>(op_expr, inputs, ctx))->at(0);
}

/* static */ Maybe<void> OpInterpUtil::Dispatch(const OpExpr& op_expr, const TensorTuple& inputs,
                                                TensorTuple* outputs,
                                                const OpExprInterpContext& ctx) {
  OF_PROFILER_RANGE_GUARD("Dispatch");
  functional::TensorProcessorPipe processor(inputs, outputs);
  if (autocast::is_enabled()) {
    JUST(processor.Apply<functional::TensorAutoCastProcessor>(
        *JUST(op_expr.GetOrCreateAutoCastMeta())));
  }
  JUST(processor.Apply<functional::TensorLayoutProcessor>(JUST(op_expr.SupportNonContiguous())));
  return JUST(GetInterpreter(processor.inputs(), ctx, op_expr))
      ->Apply(op_expr, processor.inputs(), processor.outputs(), ctx);
}

/* static */ Maybe<OpAttribute> OpInterpUtil::AddOpAndInferOpAttribute(
    const OperatorConf& op_conf, const bool is_local_strategy_enabled) {
  std::shared_ptr<OpAttribute> op_attribute = JUST([&]() -> Maybe<OpAttribute> {
    auto infer_ctx = JUST(GetCurInferCtx());
    if (is_local_strategy_enabled) {
      return infer_ctx->AddAndInferLocalOp(op_conf);
    } else {
      return infer_ctx->AddAndInferGlobalOp(op_conf);
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

}  // namespace one
}  // namespace oneflow
