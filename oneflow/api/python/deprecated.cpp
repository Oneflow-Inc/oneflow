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
#include <pybind11/pybind11.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/operator/op_attribute.pb.h"
#include "oneflow/core/operator/op_attribute.cfg.h"

#include "oneflow/core/operator/op_conf.cfg.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/framework/dtype.h"

namespace py = pybind11;

namespace oneflow {

namespace {

Maybe<cfg::OperatorConf> MakeOpConf(const std::string& serialized_str) {
  OperatorConf op_conf;
  CHECK_OR_RETURN(TxtString2PbMessage(serialized_str, &op_conf)) << "op_conf parse failed";
  return std::make_shared<cfg::OperatorConf>(op_conf);
}

Maybe<cfg::OpAttribute> MakeOpAttribute(const std::string& op_attribute_str) {
  OpAttribute op_attribute;
  CHECK_OR_RETURN(TxtString2PbMessage(op_attribute_str, &op_attribute))
      << "op_attribute parse failed";
  return std::make_shared<cfg::OpAttribute>(op_attribute);
}

Maybe<int> GetProtoDtype4OfDtype(const std::shared_ptr<DType>& x) {
  // int is the compatible data type of DType used in python code.
  return static_cast<int>(x->data_type());
}

}  // namespace

ONEFLOW_API_PYBIND11_MODULE("deprecated", m) {
  m.def("MakeOpConfByString",
        [](const std::string& str) { return MakeOpConf(str).GetPtrOrThrow(); });

  m.def("MakeOpAttributeByString",
        [](const std::string& str) { return MakeOpAttribute(str).GetPtrOrThrow(); });

  m.def("GetProtoDtype4OfDtype",
        [](const std::shared_ptr<DType>& x) { return GetProtoDtype4OfDtype(x).GetOrThrow(); });

  m.def("GetDTypeByDataType", [](int data_type) {
    return DType::GetDTypeByDataType(static_cast<DataType>(data_type)).GetPtrOrThrow();
  });
}

}  // namespace oneflow
