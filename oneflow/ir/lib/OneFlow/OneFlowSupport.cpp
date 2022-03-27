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
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "oneflow/api/common/ofblob.h"
#include "oneflow/core/common/just.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/core/functional/functional_api.yaml.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_util.h"
#include "oneflow/core/framework/user_op_registry_manager.h"
#include "oneflow/user/kernels/avg_pooling_kernel_util.h"

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

mlir::DenseElementsAttr TensorToDenseElementsAttr(
    const std::shared_ptr<::oneflow::one::Tensor>& tensor, const mlir::FloatType& float_type) {
  ::oneflow::LazyMode::Guard guard{false};
  auto shape = tensor->shape();
  std::vector<int64_t> shape_vec(shape->dim_vec().begin(), shape->dim_vec().end());
  std::vector<float> data(shape->elem_cnt());

  const auto& callback = [&](uint64_t ofblob_ptr) {
    CHECK_JUST(::oneflow::BlobBufferCopyUtil<float>::To(ofblob_ptr, data.data(), data.size()));
  };
  ::oneflow::one::SyncAccessTensorWithTimeOut(tensor, callback, "const").GetOrThrow();
  return mlir::DenseElementsAttr::get(mlir::RankedTensorType::get(shape_vec, float_type),
                                      llvm::makeArrayRef(data));
}

std::shared_ptr<::oneflow::one::Tensor> DenseElementsAttrToTensor(
    const mlir::DenseElementsAttr& attr) {
  ::oneflow::LazyMode::Guard guard{false};
  auto t = attr.getType().cast<mlir::RankedTensorType>();
  std::vector<int64_t> shape = t.getShape().vec();
  const auto device = ::oneflow::Device::ParseAndNew("cpu").GetOrThrow();
  std::shared_ptr<::oneflow::one::Tensor> tensor =
      ::oneflow::one::functional::Empty(
          ::oneflow::Shape(::oneflow::DimVector(shape.begin(), shape.end())),
          ::oneflow::DType::Get(::oneflow::DataType::kFloat).GetOrThrow(), device)
          .GetPtrOrThrow();
  std::vector<float> data(attr.getValues<float>().begin(), attr.getValues<float>().end());
  const auto& callback = [&](uint64_t of_blob_ptr) {
    ::oneflow::BlobBufferCopyUtil<float>::From(of_blob_ptr, data.data(),
                                               tensor->shape()->elem_cnt())
        .GetOrThrow();
  };
  ::oneflow::one::SyncAccessTensorWithTimeOut(tensor, callback, "mut").GetOrThrow();
  return tensor;
}

}  // namespace support

}  // namespace oneflow

}  // namespace mlir
