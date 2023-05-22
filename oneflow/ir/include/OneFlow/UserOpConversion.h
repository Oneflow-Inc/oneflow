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
#ifndef ONEFLOW_IR_INCLUDE_ONEFLOW_USEROPCONVERSION_H_
#define ONEFLOW_IR_INCLUDE_ONEFLOW_USEROPCONVERSION_H_
#include "OneFlow/OneFlowOps.h"

namespace mlir {

namespace oneflow {

namespace user_op {

::oneflow::ShapeProto getAttrAsShape(mlir::Attribute& attr);
::oneflow::Int64ListProto getAttrAsStride(mlir::Attribute& attr);
::oneflow::AttrType queryAttrType(const std::string& op_type_name, const std::string& attr_name);
LogicalResult saveAttrDictionaryToOpConf(DictionaryAttr attributes,
                                         ::oneflow::OperatorConf* op_conf);
LogicalResult ConvertUserOpAttributes(llvm::StringRef op_type_name, ValueRange operands,
                                      DictionaryAttr attributes, ::oneflow::OperatorConf& op_conf);
LogicalResult ConvertUserOpAttributes(Operation* op, ::oneflow::OperatorConf& op_conf);
LogicalResult ConvertUserOpAttributes(
    Operation* op, ::oneflow::OperatorConf& op_conf,
    bool is_mapping_size /* the input and output size should be mapped after building kernel and
                            provide information for the next query*/
    = false);
LogicalResult ConvertUserOpInputs(llvm::StringRef op_type_name, ValueRange operands,
                                  DictionaryAttr attributes, ::oneflow::UserOpConf* user_conf);
::oneflow::ParallelConf getParallelConfFromAttrDictionary(DictionaryAttr attributes);
::oneflow::ParallelConf getParallelConfFromAttrs(Attribute device_name_attr,
                                                 Attribute device_tag_attr);
::oneflow::DeviceType getDeviceTypeFromAttrDictionary(DictionaryAttr attributes);

}  // namespace user_op

}  // namespace oneflow

}  // namespace mlir

#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_USEROPCONVERSION_H_
