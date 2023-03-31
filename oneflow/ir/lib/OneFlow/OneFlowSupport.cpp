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
#include "OneFlow/OneFlowTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "oneflow/ir/include/OneFlow/OneFlowSupport.h"
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
#include "oneflow/core/common/data_type.h"

#include <iostream>
#include <vector>

namespace mlir {

namespace oneflow {

namespace support {

std::vector<std::string> GetInputKeys(const std::string& op_type_name) {
  std::vector<std::string> ret{};
  for (auto& arg : getUserOpDef(op_type_name).input()) { ret.push_back(arg.name()); }
  return ret;
}

std::vector<std::string> GetOutputKeys(const std::string& op_type_name) {
  std::vector<std::string> ret{};
  for (auto& arg : getUserOpDef(op_type_name).output()) { ret.push_back(arg.name()); }
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
          ::oneflow::DType::Get(dtype).GetOrThrow(), device, /*requires_grad=*/false,
          /*pin_memory=*/false)
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

template<typename T>
void __DenseElementsAttrToTensor(const mlir::DenseElementsAttr dense_attr,
                                 const mlir::Attribute& device_tag_attr,
                                 const mlir::Attribute& device_name_attr,
                                 const ::oneflow::DataType& dtype,
                                 std::shared_ptr<::oneflow::one::Tensor>& tensor) {
  const auto dense_type = dense_attr.getType().cast<mlir::RankedTensorType>();
  std::vector<int64_t> shape = dense_type.getShape().vec();
  int ndim = shape.size();
  CHECK_EQ(tensor->shape()->size(), ndim);
  for (int i = 0; i < ndim; ++i) { CHECK_EQ(tensor->shape()->at(i), shape[i]); }

  const auto device = MakeDevice(device_tag_attr, device_name_attr);
  CHECK(CHECK_JUST(tensor->device()) == device);

  std::vector<T> data;
  std::vector<::oneflow::float16> fp16_data;
  void* dptr = nullptr;
  const size_t tensor_size =
      tensor->shape()->elem_cnt() * ::oneflow::GetSizeOfDataType(tensor->dtype()->data_type());

  CHECK_EQ(::oneflow::GetDataType<T>::value, dtype);
  if (tensor->dtype()->data_type() == ::oneflow::DataType::kFloat16) {
    for (const T elem : dense_attr.getValues<T>()) {
      fp16_data.push_back(static_cast<::oneflow::float16>(elem));
    }
    CHECK_EQ(fp16_data.size() * sizeof(::oneflow::float16), tensor_size);
    dptr = fp16_data.data();
  } else if (tensor->dtype()->data_type() == dtype) {
    for (const T elem : dense_attr.getValues<T>()) { data.push_back(elem); }
    CHECK_EQ(data.size() * sizeof(T), tensor_size);
    dptr = data.data();
  } else {
    UNIMPLEMENTED();
  }

  const auto& callback =
      [=](::oneflow::ep::Stream* stream,
          const std::shared_ptr<::oneflow::vm::EagerBlobObject>& eager_blob_object) {
        ::oneflow::AutoMemcpy(stream, eager_blob_object->mut_dptr(), dptr, tensor_size,
                              eager_blob_object->mem_case(), ::oneflow::memory::MakeHostMemCase());
      };
  ::oneflow::one::SyncAccessTensorWithTimeOut(tensor, callback, "mut").GetOrThrow();
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

void DenseElementsAttrToTensor(const mlir::Attribute& dense_attr,
                               const mlir::Attribute& device_tag_attr,
                               const mlir::Attribute& device_name_attr,
                               std::shared_ptr<::oneflow::one::Tensor>& tensor) {
  ::oneflow::LazyMode::Guard guard{false};
  const auto dense_attr_ = dense_attr.cast<mlir::DenseElementsAttr>();
  const auto dense_element_type = dense_attr_.getElementType();
  if (dense_element_type.isF32()) {
    __DenseElementsAttrToTensor<float>(dense_attr_, device_tag_attr, device_name_attr,
                                       ::oneflow::DataType::kFloat, tensor);
  } else {
    llvm::errs() << "Converting mlir::DenseElementsAttr to oneflow::Tensor only support float32 "
                    "and int64 now."
                 << "\n";
    exit(EXIT_FAILURE);
  }
}

FailureOr<::oneflow::DataType> FromMLIRTypeToOFDataType(Type mlir_type) {
  if (mlir_type.dyn_cast<InvalidElementType>()) { return ::oneflow::DataType::kInvalidDataType; }
  if (mlir_type.dyn_cast<CharElementType>()) { return ::oneflow::DataType::kChar; }
  if (mlir_type.dyn_cast<OFRecordElementType>()) { return ::oneflow::DataType::kOFRecord; }
  if (mlir_type.dyn_cast<TensorBufferElementType>()) { return ::oneflow::DataType::kTensorBuffer; }
  if (mlir_type.isF16()) { return ::oneflow::DataType::kFloat16; }
  if (mlir_type.isF32()) { return ::oneflow::DataType::kFloat; }
  if (mlir_type.isF64()) { return ::oneflow::DataType::kDouble; }

  if (mlir_type.isSignlessInteger(8)) { return ::oneflow::DataType::kBool; }
  if (mlir_type.isSignlessInteger(16)) { return ::oneflow::DataType::kUInt16; }
  if (mlir_type.isSignlessInteger(32)) { return ::oneflow::DataType::kUInt32; }
  if (mlir_type.isSignlessInteger(64)) { return ::oneflow::DataType::kUInt64; }
  if (mlir_type.isSignlessInteger(128)) { return ::oneflow::DataType::kUInt128; }

  if (mlir_type.isSignedInteger(8)) { return ::oneflow::DataType::kInt8; }
  if (mlir_type.isSignedInteger(16)) { return ::oneflow::DataType::kInt16; }
  if (mlir_type.isSignedInteger(32)) { return ::oneflow::DataType::kInt32; }
  if (mlir_type.isSignedInteger(64)) { return ::oneflow::DataType::kInt64; }
  if (mlir_type.isSignedInteger(128)) { return ::oneflow::DataType::kInt128; }
  llvm::errs() << "Unsupported data type: " << mlir_type << "\n";
  return failure();
}

FailureOr<::oneflow::DataType> FromMLIRDataTypeToOFDataType(::mlir::oneflow::DataType data_type) {
  switch (data_type) {
    case ::mlir::oneflow::DataType::DT_InvalidDataType:
      return ::oneflow::DataType::kInvalidDataType;
#define DEFINE_ONE_CASE(datatype) \
  case ::mlir::oneflow::DataType::DT_##datatype: return ::oneflow::DataType::k##datatype;
      DEFINE_ONE_CASE(Char)
      DEFINE_ONE_CASE(Float)
      DEFINE_ONE_CASE(Double)
      DEFINE_ONE_CASE(Int8)
      DEFINE_ONE_CASE(Int32)
      DEFINE_ONE_CASE(Int64)
      DEFINE_ONE_CASE(UInt8)
      DEFINE_ONE_CASE(OFRecord)
      DEFINE_ONE_CASE(Float16)
      DEFINE_ONE_CASE(TensorBuffer)
      DEFINE_ONE_CASE(Bool)
#undef DEFINE_ONE_CASE
    default: {
      return failure();
    }
  }
  return failure();
}

FailureOr<::oneflow::DataType> FromMLIRAttrToOFDataType(Attribute attr) {
  const auto data_type_attr = attr.dyn_cast<mlir::oneflow::DataTypeAttr>();
  return FromMLIRDataTypeToOFDataType(data_type_attr.getValue());
}

const ::oneflow::UserOpDef& getUserOpDef(const std::string& op_type_name) {
  const ::oneflow::user_op::OpRegistryResult* val =
      ::oneflow::user_op::UserOpRegistryMgr::Get().GetOpRegistryResult(op_type_name);
  CHECK(val) << " Cannot find op_type_name: " << op_type_name;
  return val->op_def;
}

}  // namespace support

}  // namespace oneflow

}  // namespace mlir
