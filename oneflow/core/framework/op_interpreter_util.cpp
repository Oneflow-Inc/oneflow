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
#include "oneflow/core/framework/op_interpreter_util.h"

#include "oneflow/core/eager/foreign_boxing_util.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/framework/py_distribute.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {
namespace one {

namespace {

std::shared_ptr<OpExprInterpreter> BuildInterpreter(const bool& eager_mode) {
  std::shared_ptr<NormalInterpreter> normal_interp;
  if (eager_mode) {
    normal_interp = std::make_shared<EagerInterpreter>();
  } else {
    normal_interp = std::make_shared<LazyInterpreter>();
  }
  return std::make_shared<AutogradInterpreter>(normal_interp);
}

}  // namespace

/*static*/ Maybe<OpExprInterpreter> OpInterpUtil::GetInterpreter() {
  static const auto& g_lazy_interpreter = BuildInterpreter(false);
  static const auto& g_eager_interpreter = BuildInterpreter(true);
  if (EagerExecutionEnabled()) { return g_eager_interpreter; }
  return g_lazy_interpreter;
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

/*static*/ Maybe<cfg::OpAttribute> OpInterpUtil::InferOpAttribute(const BuiltinOpExpr& op_expr,
                                                                  const TensorTuple& inputs) {
  const auto& scope = JUST(GetCurrentScope());
  auto op_conf = JUST(GenBuiltinOpConf(op_expr));
  int64_t symbol_id = JUST(scope->symbol_id());
  op_conf->set_scope_symbol_id(symbol_id);
  if (!op_conf->has_device_tag()) {
    op_conf->set_device_tag(scope->device_parallel_desc_symbol()->device_tag());
  }
  const auto& ibn2blob_object = JUST(MakeBn2BlobObjectMap(op_expr.indexed_ibns(), inputs));
  OpNodeSignature upstream_signature;
  if (ibn2blob_object->size()) {
    std::shared_ptr<cfg::OpNodeSignature> cfg_upstream_signature(new cfg::OpNodeSignature);
    for (const auto& it : *ibn2blob_object) {
      it.second->op_arg_parallel_attr()->DumpToOpNodeSignature(it.first, cfg_upstream_signature);
      it.second->op_arg_blob_attr()->DumpToOpNodeSignature(it.first, cfg_upstream_signature);
    }
    cfg_upstream_signature->ToProto(&upstream_signature);
  }
  const auto& op = JUST(ConstructAndInferOp(*op_conf, upstream_signature, *scope));
  const auto& op_attribute = op->GetOpAttributeWithoutOpNameAndLbn();
  return std::make_shared<cfg::OpAttribute>(*op_attribute);
}

using Bn2BlobObjectMap = HashMap<std::string, std::shared_ptr<compatible_py::BlobObject>>;

/*static*/ Maybe<Bn2BlobObjectMap> OpInterpUtil::MakeBn2BlobObjectMap(
    const std::vector<std::string>& indexed_ibns, const TensorTuple& inputs) {
  CHECK_EQ_OR_RETURN(indexed_ibns.size(), inputs.size());
  auto bn2blob_object = std::make_shared<Bn2BlobObjectMap>();
  for (int i = 0; i < inputs.size(); ++i) {
    const auto& ibn = indexed_ibns.at(i);
    const auto& blob_object = JUST(GetTensorBlobObject(inputs[i]));
    bn2blob_object->emplace(ibn, blob_object);
  }
  return bn2blob_object;
}

/*static*/ Maybe<OperatorConf> OpInterpUtil::GenBuiltinOpConf(const BuiltinOpExpr& op_expr) {
  auto op_conf = std::make_shared<OperatorConf>();
  op_expr.BuildOpConf(op_conf.get());
  return op_conf;
}

/*static*/ Maybe<compatible_py::BlobObject> OpInterpUtil::GetTensorBlobObject(
    const std::shared_ptr<Tensor>& tensor) {
  if (auto* mirrored_tensor = dynamic_cast<MirroredTensor*>(tensor.get())) {
    return mirrored_tensor->blob_object();
  } else if (auto* consistent_tensor = dynamic_cast<ConsistentTensor*>(tensor.get())) {
    return consistent_tensor->blob_object();
  } else {
    UNIMPLEMENTED_THEN_RETURN()
        << "The tensor should be either Mirrored Tensor or Consistent Tensor.";
  }
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
        blob_attr->shape(), dtype, device, is_lazy, /*requires_grad=*/false, /*is_leaf=*/false,
        /*retain_grad=*/false));
  } else {
    const auto& distribute =
        compatible_py::MakeDistribute(*(parallel_attr->sbp_parallel())).GetPtrOrThrow();
    return static_cast<std::shared_ptr<Tensor>>(ConsistentTensor::MakeTensor(
        blob_attr->shape(), dtype, distribute, parallel_attr->parallel_desc_symbol(), is_lazy,
        /*requires_grad=*/false, /*is_leaf=*/false, /*retain_grad=*/false));
  }
}

/*static*/ Maybe<Tensor> OpInterpUtil::BuildTensorFromBlobObject(
    const std::shared_ptr<compatible_py::BlobObject>& blob_object) {
  const auto& blob_attr = blob_object->op_arg_blob_attr();
  const auto& parallel_attr = blob_object->op_arg_parallel_attr();
  const auto& tensor = JUST(BuildTensor(blob_attr, parallel_attr, /*is_lazy=*/false));
  if (parallel_attr->is_mirrored()) {
    dynamic_cast<MirroredTensor*>(tensor.get())->set_blob_object(blob_object);
  } else {
    dynamic_cast<ConsistentTensor*>(tensor.get())->set_blob_object(blob_object);
  }
  return tensor;
}

}  // namespace one
}  // namespace oneflow
