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

#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/eager/foreign_boxing_util.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/framework/py_distribute.h"
#include "oneflow/core/framework/tensor_impl.h"
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

}  // namespace

/*static*/ Maybe<AutogradInterpreter> OpInterpUtil::GetInterpreter() {
  static const auto& g_lazy_interpreter = BuildLazyInterpreter();
  static const auto& g_eager_consistent_interpreter = BuildEagerInterpreter(/*is_mirrored=*/false);
  static const auto& g_eager_mirrored_interpreter = BuildEagerInterpreter(/*is_mirrored=*/true);
  if (EagerExecutionEnabled()) {
    const auto& session = JUST(GetDefaultSession());
    bool is_mirrored_strategy_enabled = session->is_mirrored_strategy_enabled_stack()->empty()
                                        || JUST(session->IsMirroredStrategyEnabled());
    if (is_mirrored_strategy_enabled) {
      return g_eager_mirrored_interpreter;
    } else {
      return g_eager_consistent_interpreter;
    }
  }
  return g_lazy_interpreter;
}

template<>
/*static*/ Maybe<TensorTuple> OpInterpUtil::Dispatch<TensorTuple>(const OpExpr& op_expr,
                                                                  const TensorTuple& inputs,
                                                                  const AttrMap& attrs) {
  auto outputs = std::make_shared<TensorTuple>(op_expr.output_size());
  JUST(GetInterpreter())->Apply(op_expr, inputs, outputs.get(), attrs);
  return outputs;
}

template<>
/*static*/ Maybe<Tensor> OpInterpUtil::Dispatch<Tensor>(const OpExpr& op_expr,
                                                        const TensorTuple& inputs,
                                                        const AttrMap& attrs) {
  return JUST(Dispatch<TensorTuple>(op_expr, inputs, attrs))->at(0);
}

/*static*/ Maybe<cfg::OpAttribute> OpInterpUtil::AddOpAndInferOpAttribute(
    const OperatorConf& op_conf, const bool is_mirrored_strategy_enabled) {
  std::shared_ptr<OpAttribute> op_attribute = JUST([&]() -> Maybe<OpAttribute> {
    auto infer_ctx = JUST(GetCurInferCtx());
    if (is_mirrored_strategy_enabled) {
      return infer_ctx->AddAndInferMirroredOp(op_conf);
    } else {
      return infer_ctx->AddAndInferConsistentOp(op_conf);
    }
  }());
  return std::make_shared<cfg::OpAttribute>(*op_attribute);
}

/*static*/ Maybe<OperatorConf> OpInterpUtil::GenBuiltinOpConf(const BuiltinOpExpr& op_expr,
                                                              const AttrMap& attrs) {
  auto op_conf = std::make_shared<OperatorConf>();
  op_expr.BuildOpConf(op_conf.get(), attrs);
  return op_conf;
}

/*static*/ Maybe<Tensor> OpInterpUtil::BuildTensor(
    const std::shared_ptr<compatible_py::OpArgBlobAttribute>& blob_attr,
    const std::shared_ptr<compatible_py::OpArgParallelAttribute>& parallel_attr,
    const bool is_lazy) {
  const auto& dtype = JUST(DType::GetDTypeByDataType(DataType(blob_attr->get_dtype())));
  if (parallel_attr->is_mirrored()) {
    const auto& device =
        JUST(Device::MakeDeviceByParallelDesc(*parallel_attr->parallel_desc_symbol()));
    return static_cast<std::shared_ptr<Tensor>>(MirroredTensor::MakeTensor(
        blob_attr->shape(), dtype, device, is_lazy, /*requires_grad=*/false, /*is_leaf=*/false));
  } else {
    const auto& distribute = JUST(compatible_py::MakeDistribute(*(parallel_attr->sbp_parallel())));
    return static_cast<std::shared_ptr<Tensor>>(ConsistentTensor::MakeTensor(
        blob_attr->shape(), dtype, distribute, parallel_attr->parallel_desc_symbol(), is_lazy,
        /*requires_grad=*/false, /*is_leaf=*/false));
  }
}

/*static*/ Maybe<Tensor> OpInterpUtil::BuildEagerMirroredTensorFromEagerBlobObject(
    const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object,
    const std::shared_ptr<const Device>& device) {
  auto tensor = MirroredTensor::MakeEagerTensor(eager_blob_object, device,
                                                /* requires_grad */ false, /* is_leaf */ false);
  return std::static_pointer_cast<Tensor>(tensor);
}

}  // namespace one
}  // namespace oneflow
