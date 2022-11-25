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
// this file should contains functions to get operands and results with user op name and index

#include "OneFlow/UserOpConversion.h"
#include "OneFlow/UserOpReflection.h"
#include "oneflow/core/framework/user_op_def.h"

namespace mlir {

namespace oneflow {

namespace user_op {

namespace {

LogicalResult saveAttrToOpConf(DictionaryAttr attributes, ::oneflow::OperatorConf* op_conf) {
  if (auto scope_symbol_id =
          attributes.get(OpTrait::IsOpConfCompatible<void>::getScopeSymbolIDAttr())
              .dyn_cast_or_null<IntegerAttr>()) {
    op_conf->set_scope_symbol_id(scope_symbol_id.getInt());
  }
  if (auto op_name = attributes.get(OpTrait::IsOpConfCompatible<void>::getOpNameAttr())
                         .dyn_cast_or_null<StringAttr>()) {
    op_conf->set_name(op_name.str());
  }
  op_conf->set_device_tag(attributes.get(OpTrait::IsOpConfCompatible<void>::getDeviceTagAttr())
                              .cast<StringAttr>()
                              .str());
  ;
  return success();
}

void WriteAttrToShape(mlir::Attribute& attr, ::oneflow::ShapeProto* shape) {
  for (auto v : attr.dyn_cast<ArrayAttr>().getValue()) {
    shape->add_dim(v.dyn_cast<IntegerAttr>().getSInt());
  }
}

void WriteAttrToStride(mlir::Attribute& attr, ::oneflow::Int64ListProto* stride) {
  for (auto v : attr.dyn_cast<ArrayAttr>().getValue()) {
    stride->add_dim(v.dyn_cast<IntegerAttr>().getSInt());
  }
}

const ::oneflow::UserOpDef& GetUserOpDef(const std::string& op_type_name) {
  const ::oneflow::user_op::OpRegistryResult* val =
      ::oneflow::user_op::UserOpRegistryMgr::Get().GetOpRegistryResult(op_type_name);
  CHECK(val) << " Cannot find op_type_name: " << op_type_name;
  return val->op_def;
}

::oneflow::AttrType QueryAttrType(const std::string& op_type_name, const std::string& attr_name) {
  ::oneflow::user_op::UserOpDefWrapper op_def(GetUserOpDef(op_type_name));
  CHECK(op_def.IsAttrName(attr_name)) << attr_name << " not a attr name for op: " << op_type_name;
  return op_def.GetAttrType(attr_name);
}

}  // namespace

LogicalResult ConvertUserOpAttributes(std::string& op_type_name, DictionaryAttr attributes,
                                      ::oneflow::OperatorConf& op_conf) {
  auto user_conf = op_conf.mutable_user_conf();
  op_conf.mutable_user_conf()->set_op_type_name(op_type_name);
  CHECK(saveAttrToOpConf(attributes, &op_conf).succeeded());
  for (auto id_attr : attributes) {
    auto id = id_attr.getName();
    // mlir only attrs
    // TODO: prefix special attributes with "oneflow.". For example: `oneflow.op_type_name = "add"`
    if (id.strref().equals("callee")
        || id.strref().equals(OpTrait::IsOpConfCompatible<void>::getDeviceNameAttr())
        || id.strref().equals(OpTrait::IsOpConfCompatible<void>::getHierarchyAttr())
        || id.strref().equals(OpTrait::IsImportCompatible<void>::getOutputLBNsAttr())
        || id.strref().equals(OpTrait::IsAlternative<void>::getOpTypeNameAttr())
        || id.strref().equals(
            mlir::OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr())
        || id.strref().equals(
            mlir::OpTrait::AttrSizedResultSegments<void>::getResultSegmentSizeAttr())) {
      continue;
    } else if (id.strref().equals("input_sizes") || id.strref().equals("output_sizes")) {
      continue;
    }
    // convert op conf attributes
    else if (id.strref().equals(OpTrait::IsOpConfCompatible<void>::getOpNameAttr())) {
      continue;
    } else if (id.strref().equals(OpTrait::IsOpConfCompatible<void>::getDeviceTagAttr())) {
      continue;
    } else if (id.strref().equals(OpTrait::IsOpConfCompatible<void>::getScopeSymbolIDAttr())) {
      continue;
    }
    // convert user conf attributes
    else {
      auto attr_name = id.str();
      Attribute attr = id_attr.getValue();
      auto user_attr = ::oneflow::AttrValue();
      const ::oneflow::AttrType attr_type = QueryAttrType(op_type_name, attr_name);
      if (attr_type == ::oneflow::kAtInt32) {
        user_attr.set_at_int32(attr.dyn_cast<IntegerAttr>().getSInt());
      } else if (attr_type == ::oneflow::kAtInt64) {
        user_attr.set_at_int64(attr.dyn_cast<IntegerAttr>().getSInt());
      } else if (attr_type == ::oneflow::kAtBool) {
        user_attr.set_at_bool(attr.dyn_cast<BoolAttr>().getValue());
      } else if (attr_type == ::oneflow::kAtFloat) {
        user_attr.set_at_float(attr.dyn_cast<FloatAttr>().getValue().convertToFloat());
      } else if (attr_type == ::oneflow::kAtDouble) {
        user_attr.set_at_double(attr.dyn_cast<FloatAttr>().getValue().convertToDouble());
      } else if (attr_type == ::oneflow::kAtString) {
        user_attr.set_at_string(attr.dyn_cast<StringAttr>().getValue().str());
      } else if (attr_type == ::oneflow::kAtShape) {
        WriteAttrToShape(attr, user_attr.mutable_at_shape());
      } else if (attr_type == ::oneflow::kAtStride) {
        WriteAttrToStride(attr, user_attr.mutable_at_stride());
      } else if (attr_type == ::oneflow::kAtDataType) {
        const auto dt = support::FromMLIRAttrToOFDataType(attr);
        if (succeeded(dt)) {
          user_attr.set_at_data_type(dt.getValue());
        } else {
          LOG(FATAL) << "fail to convert op attr to data type, key: " + id.str();
          return failure();
        }
      } else if (attr_type == ::oneflow::kAtListInt32) {
        user_attr.mutable_at_list_int32();
        auto ref = attr.dyn_cast<ArrayAttr>();
        for (auto v : ref.getValue()) {
          user_attr.mutable_at_list_int32()->add_val(v.dyn_cast<IntegerAttr>().getSInt());
        }
      } else if (attr_type == ::oneflow::kAtListInt64) {
        user_attr.mutable_at_list_int64();
        auto ref = attr.dyn_cast<ArrayAttr>();
        for (auto v : ref.getValue()) {
          user_attr.mutable_at_list_int64()->add_val(v.dyn_cast<IntegerAttr>().getSInt());
        }
      } else if (attr_type == ::oneflow::kAtListFloat) {
        user_attr.mutable_at_list_float();
        auto ref = attr.dyn_cast<ArrayAttr>();
        for (auto v : ref.getValue()) {
          user_attr.mutable_at_list_float()->add_val(
              v.dyn_cast<FloatAttr>().getValue().convertToFloat());
        }
      } else if (attr_type == ::oneflow::kAtListDataType) {
        for (auto v : attr.dyn_cast<ArrayAttr>().getValue()) {
          const auto dt = support::FromMLIRAttrToOFDataType(attr);
          if (succeeded(dt)) {
            user_attr.mutable_at_list_data_type()->add_val(dt.getValue());
          } else {
            LOG(FATAL) << "fail to convert op attr to data type, key: " + id.str();
            return failure();
          }
        }
      } else if (attr_type == ::oneflow::kAtListShape) {
        for (auto shape_attr : attr.dyn_cast<ArrayAttr>().getValue()) {
          ::oneflow::ShapeProto* shape_ptr = user_attr.mutable_at_list_shape()->add_val();
          WriteAttrToShape(shape_attr, shape_ptr);
        }
      } else if (attr_type == ::oneflow::kAtListStride) {
        for (auto stride_attr : attr.dyn_cast<ArrayAttr>().getValue()) {
          ::oneflow::Int64ListProto* stride_ptr = user_attr.mutable_at_list_stride()->add_val();
          WriteAttrToStride(stride_attr, stride_ptr);
        }
      } else if (attr_type == ::oneflow::kAtListString) {
        // attr like nd_sbp requires the existence of list even it is empty
        user_attr.mutable_at_list_string();
        for (auto s : attr.dyn_cast<ArrayAttr>().getValue()) {
          user_attr.mutable_at_list_string()->add_val(s.dyn_cast<StringAttr>().getValue().str());
        }
      } else {
        return failure();
      }
      (*user_conf->mutable_attr())[id.str()] = user_attr;
    }
  }
  return success();
}

LogicalResult ConvertUserOpAttributes(Operation* op, ::oneflow::OperatorConf& op_conf) {
  std::string op_type_name = GetOpTypeName(op);
  {
    std::vector<std::string> keys{};
    std::vector<int32_t> sizes{};
    if (failed(user_op::GetFilteredSegmentKeyAndSizes<OpTrait::AttrSizedOperandSegments>(op, keys,
                                                                                         sizes))) {
      op->emitError("fail to convert user op input order");
      return failure();
    }
    for (const auto& s : keys) { op_conf.mutable_user_conf()->add_input_order(s); }
  }
  {
    std::vector<std::string> keys{};
    std::vector<int32_t> sizes{};
    if (failed(user_op::GetFilteredSegmentKeyAndSizes<OpTrait::AttrSizedResultSegments>(op, keys,
                                                                                        sizes))) {
      op->emitError("fail to convert user op output order");
      return failure();
    }
    for (const auto& s : keys) { op_conf.mutable_user_conf()->add_output_order(s); }
  }
  return ConvertUserOpAttributes(op_type_name, op->getAttrDictionary(), op_conf);
}

}  // namespace user_op

}  // namespace oneflow

}  // namespace mlir
