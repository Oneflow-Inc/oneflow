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
#include "OneFlow/UserOpConversion.h"
#include "OneFlow/UserOpReflection.h"
#include "oneflow/core/framework/infer_util.h"
#include "oneflow/core/framework/user_op_attr.pb.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/kernel/blob_tensor_view.h"
#include "oneflow/core/memory/memory_case.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "OneFlow/OKL/Kernel/InferContext.h"
#include "OneFlow/OKL/Kernel/RegContext.h"
#include "oneflow/core/framework/user_op_kernel_registry.h"
#include "oneflow/ir/oneflow-translate/include/OneFlow/MLIROneFlowTranslation.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"

namespace oneflow {
namespace okl {

static user_op::UserOpConfWrapper GetConfWrapper(mlir::Operation* op,
                                                 bool is_mapping_size = false) {
  OperatorConf op_conf;
  if (mlir::failed(mlir::oneflow::user_op::ConvertUserOpAttributes(op, op_conf, is_mapping_size))) {
    op->emitError("fail to convert user op attributes");
    exit(1);
  }
  auto conf_wrapper_ = user_op::UserOpConfWrapper(std::make_shared<OperatorConf>(op_conf));
  return conf_wrapper_;
}

RegContext::RegContext(mlir::Operation* op) : op_(op), conf_wrapper_(GetConfWrapper(op, true)) {
  const auto handle_operands_or_results =
      [&op, this](const auto& arg_ids, const auto& get_operand_or_result, ArgVec& arg_vec) {
        for (const auto& obj_id : ::llvm::enumerate(arg_ids)) {
          user_op::NaiveTensorDesc tensor_desc{};
          auto obj = get_operand_or_result(op, obj_id.index());
          if (auto rankedTensorType = obj.getType().template dyn_cast<mlir::RankedTensorType>()) {
            tensor_desc.set_shape(
                Shape{rankedTensorType.getShape().begin(), rankedTensorType.getShape().end()});
            const auto data_type =
                mlir::oneflow::support::FromMLIRTypeToOFDataType(rankedTensorType.getElementType());
            if (mlir::failed(data_type)) { exit(1); }
            tensor_desc.set_data_type(data_type.getValue());
            llvm::SmallVector<int64_t> strides;
            int64_t _;
            auto mem_type = mlir::MemRefType::get(rankedTensorType.getShape(),
                                                  rankedTensorType.getElementType());
            if (failed(mlir::getStridesAndOffset(mem_type, strides, _))) {
              LOG(FATAL) << "Fail to get stride from memory type";
            }
            tensor_desc.set_stride(Stride(strides.begin(), strides.end()));
            // TODO: set is_dynamic
          } else {
            LOG(FATAL) << "Unranked tensor type not supported";
          }
          CHECK(arg2tensor_desc_.emplace(obj_id.value(), tensor_desc).second) << "duplicate key";
          arg_vec.push_back(obj_id.value());
        }
      };
  handle_operands_or_results(
      ::mlir::oneflow::user_op::ArgIds<mlir::OpTrait::AttrSizedOperandSegments>(op),
      [](auto& x, size_t index) { return x->getOperand(index); }, inputs_);
  handle_operands_or_results(
      ::mlir::oneflow::user_op::ArgIds<mlir::OpTrait::AttrSizedResultSegments>(op),
      [](auto& x, size_t index) { return x->getResult(index); }, outputs_);

  auto dev_tag = mlir::OpTrait::IsOpConfCompatible<void>::getDeviceTag(op);
  if (dev_tag == "cpu") {
    device_type_ = DeviceType::kCPU;
  } else if (dev_tag == "cuda") {
    device_type_ = DeviceType::kCUDA;
  } else {
    LOG(FATAL) << "Unsupported device tag: " << dev_tag.str();
  }
  auto op_name = GetOp()->getName().stripDialect().str();
  if (const auto op_type_name =
          GetOp()->getAttr("op_type_name").dyn_cast_or_null<mlir::StringAttr>()) {
    op_name = op_type_name.str();
  }

  reg_res_ =
      CHECK_JUST(user_op::UserOpRegistryMgr::Get().GetOpKernelRegistryResult(op_name, *this));
  kernel_ = reg_res_->create_fn();

  conf_wrapper_ = GetConfWrapper(op_, true);
}

DeviceType RegContext::device_type() const { return device_type_; }
const ParallelContext& RegContext::parallel_ctx() const {
  TODO() << "create parallel_ctx from op in mlir";
  ParallelContext* parallel_ctx = nullptr;
  return *parallel_ctx;
}
const user_op::TensorDesc* RegContext::TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                                  int32_t index) const {
  auto it = arg2tensor_desc_.find(std::make_pair(arg_name, index));
  if (it == arg2tensor_desc_.end()) { return nullptr; }
  return &(it->second);
}
const ArgVec& RegContext::inputs() const { return inputs_; }
const ArgVec& RegContext::outputs() const { return outputs_; }

// TODO: more information is needed
const user_op::UserOpConfWrapper& RegContext::user_op_conf() const { return conf_wrapper_; }

const std::shared_ptr<const user_op::AttrVal>& RegContext::Attr4Name(
    const std::string& attr_name) const {
  return user_op_conf().Attr4Name(attr_name);
}

const size_t RegContext::GetTmpBufferSize() const {
  if (reg_res_->need_temp_storage) {
    InferContext infer_ctx(this);
    return reg_res_->infer_tmp_size_fn(&infer_ctx);
  }
  return 0;
}

}  // namespace okl
}  // namespace oneflow
