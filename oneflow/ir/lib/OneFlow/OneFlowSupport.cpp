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
#include <iostream>
#include <vector>
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "mlir/IR/MLIRContext.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/just.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/core/functional/functional_api.yaml.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_util.h"
#include "oneflow/core/framework/user_op_registry_manager.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/memory/memory_case_util.h"

namespace mlir {

namespace oneflow {

namespace support {

using ::oneflow::UserOpDef;
using ::oneflow::user_op::OpRegistryResult;
using ::oneflow::user_op::UserOpRegistryMgr;

const UserOpDef& GetUserOpDef(const std::string& op_type_name) {
  const OpRegistryResult* val = UserOpRegistryMgr::Get().GetOpRegistryResult(op_type_name);
  CHECK(val) << " Cannot find op_type_name: " << op_type_name;
  return val->op_def;
}

std::vector<std::string> GetInputKeys(const std::string& op_type_name) {
  std::vector<std::string> ret{};
  for (auto& arg : GetUserOpDef(op_type_name).input()) { ret.push_back(arg.name()); }
  return ret;
}

std::vector<std::string> GetOutputKeys(const std::string& op_type_name) {
  std::vector<std::string> ret{};
  for (auto& arg : GetUserOpDef(op_type_name).output()) { ret.push_back(arg.name()); }
  return ret;
}

namespace {

::oneflow::Symbol<::oneflow::Device> MakeDevice(const mlir::Attribute& device_tag_attr,
                                                const mlir::Attribute& device_name_attr) {
  const auto device_tag = device_tag_attr.cast<mlir::StringAttr>().str();
  const auto device_name =
      device_name_attr.cast<mlir::ArrayAttr>().getValue().front().cast<mlir::StringAttr>().str();
  const std::string device_info =
      device_tag == "gpu" ? "cuda" : device_tag + device_name.substr(device_name.rfind(":"));
  return ::oneflow::Device::ParseAndNew(device_info).GetOrThrow();
}

template<typename T, typename MLIR_T>
mlir::DenseElementsAttr __TensorToDenseElementsAttr(
    const std::shared_ptr<::oneflow::one::Tensor>& tensor, const MLIR_T& mlir_type) {
  ::oneflow::LazyMode::Guard guard{false};
  const auto tensor_ = ::oneflow::one::functional::ToContiguous(tensor).GetPtrOrThrow();
  auto shape = tensor_->shape();
  std::vector<int64_t> shape_vec(shape->dim_vec().begin(), shape->dim_vec().end());
  std::vector<T> data(shape->elem_cnt());
  const auto& callback =
      [&](::oneflow::ep::Stream* stream,
          const std::shared_ptr<::oneflow::vm::EagerBlobObject>& eager_blob_object) {
        ::oneflow::AutoMemcpy(stream, data.data(), eager_blob_object->dptr(),
                              data.size() * sizeof(T), ::oneflow::memory::MakeHostMemCase(),
                              eager_blob_object->mem_case());
      };
  ::oneflow::one::SyncAccessTensorWithTimeOut(tensor_, callback, "const").GetOrThrow();
  return mlir::DenseElementsAttr::get(mlir::RankedTensorType::get(shape_vec, mlir_type),
                                      llvm::makeArrayRef(data));
}

template<typename T>
std::shared_ptr<::oneflow::one::Tensor> __DenseElementsAttrToTensor(
    const mlir::DenseElementsAttr dense_attr, const mlir::Attribute& device_tag_attr,
    const mlir::Attribute& device_name_attr, const ::oneflow::DataType& dtype) {
  const auto dense_type = dense_attr.getType().cast<mlir::RankedTensorType>();
  std::vector<int64_t> shape = dense_type.getShape().vec();

  const auto device = MakeDevice(device_tag_attr, device_name_attr);

  std::shared_ptr<::oneflow::one::Tensor> tensor =
      ::oneflow::one::functional::Empty(
          ::oneflow::Shape(::oneflow::DimVector(shape.begin(), shape.end())),
          ::oneflow::DType::Get(dtype).GetOrThrow(), device, /*pin_memory=*/false)
          .GetPtrOrThrow();

  std::vector<T> data(dense_attr.getValues<T>().begin(), dense_attr.getValues<T>().end());
  const auto& callback =
      [&](::oneflow::ep::Stream* stream,
          const std::shared_ptr<::oneflow::vm::EagerBlobObject>& eager_blob_object) {
        ::oneflow::AutoMemcpy(stream, eager_blob_object->mut_dptr(), data.data(),
                              tensor->shape()->elem_cnt() * sizeof(T),
                              eager_blob_object->mem_case(), ::oneflow::memory::MakeHostMemCase());
      };
  ::oneflow::one::SyncAccessTensorWithTimeOut(tensor, callback, "mut").GetOrThrow();
  return tensor;
}

}  // namespace

mlir::DenseElementsAttr TensorToDenseElementsAttr(
    const std::shared_ptr<::oneflow::one::Tensor>& tensor, MLIRContext* ctx) {
  const auto dtype = tensor->dtype()->data_type();
  if (dtype == ::oneflow::DataType::kFloat) {
    return __TensorToDenseElementsAttr<float, mlir::FloatType>(tensor,
                                                               mlir::FloatType::getF32(ctx));
  } else if (dtype == ::oneflow::DataType::kInt64) {
    auto mlir_type = mlir::IntegerType::IntegerType::get(
        ctx, 64, mlir::IntegerType::SignednessSemantics::Signed);
    return __TensorToDenseElementsAttr<int64_t, mlir::IntegerType>(tensor, mlir_type);
  }
  llvm::errs() << "Converting oneflow::Tensor to mlir::DenseElementsAttr only support float32 now."
               << "\n";
  exit(EXIT_FAILURE);
}

std::shared_ptr<::oneflow::one::Tensor> DenseElementsAttrToTensor(
    const mlir::Attribute& dense_attr, const mlir::Attribute& device_tag_attr,
    const mlir::Attribute& device_name_attr) {
  ::oneflow::LazyMode::Guard guard{false};
  const auto dense_attr_ = dense_attr.cast<mlir::DenseElementsAttr>();
  const auto dense_element_type = dense_attr_.getElementType();
  if (dense_element_type.isF32()) {
    return __DenseElementsAttrToTensor<float>(dense_attr_, device_tag_attr, device_name_attr,
                                              ::oneflow::DataType::kFloat);
  }
  llvm::errs()
      << "Converting mlir::DenseElementsAttr to oneflow::Tensor only support float32 and int64 now."
      << "\n";
  exit(EXIT_FAILURE);
}

}  // namespace support

}  // namespace oneflow

}  // namespace mlir
