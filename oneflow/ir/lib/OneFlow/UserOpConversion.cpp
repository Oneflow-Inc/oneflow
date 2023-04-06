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

LogicalResult saveAttrDictionaryToOpConf(DictionaryAttr attributes,
                                         ::oneflow::OperatorConf* op_conf) {
  if (auto scope_symbol_id =
          attributes.get(OpTrait::IsOpConfCompatible<void>::getScopeSymbolIDAttr())
              .dyn_cast_or_null<IntegerAttr>()) {
    op_conf->set_scope_symbol_id(scope_symbol_id.getInt());
  }
  if (auto op_name = attributes.get(OpTrait::IsOpConfCompatible<void>::getOpNameAttr())
                         .dyn_cast_or_null<StringAttr>()) {
    op_conf->set_name(op_name.str());
  }
  auto device_tag = attributes.get(OpTrait::IsOpConfCompatible<void>::getDeviceTagAttr())
                        .dyn_cast_or_null<StringAttr>();
  CHECK(device_tag) << "attr absent: "
                    << OpTrait::IsOpConfCompatible<void>::getDeviceTagAttr().str();
  op_conf->set_device_tag(device_tag.str());
  return success();
}

LogicalResult doConvertUserOpAttributes(llvm::StringRef op_type_name, DictionaryAttr attributes,
                                        ::oneflow::OperatorConf& op_conf) {
  auto user_conf = op_conf.mutable_user_conf();
  op_conf.mutable_user_conf()->set_op_type_name(op_type_name.str());
  CHECK(saveAttrDictionaryToOpConf(attributes, &op_conf).succeeded());
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
      const ::oneflow::AttrType attr_type = queryAttrType(op_type_name.str(), attr_name);
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
        *user_attr.mutable_at_shape() = getAttrAsShape(attr);
      } else if (attr_type == ::oneflow::kAtStride) {
        *user_attr.mutable_at_stride() = getAttrAsStride(attr);
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
          *shape_ptr = getAttrAsShape(shape_attr);
        }
      } else if (attr_type == ::oneflow::kAtListStride) {
        for (auto stride_attr : attr.dyn_cast<ArrayAttr>().getValue()) {
          ::oneflow::Int64ListProto* stride_ptr = user_attr.mutable_at_list_stride()->add_val();
          *stride_ptr = getAttrAsStride(stride_attr);
        }
      } else if (attr_type == ::oneflow::kAtListString) {
        // attr like nd_sbp requires the existence of list even it is empty
        user_attr.mutable_at_list_string();
        for (auto s : attr.dyn_cast<ArrayAttr>().getValue()) {
          user_attr.mutable_at_list_string()->add_val(s.dyn_cast<StringAttr>().getValue().str());
        }
      } else if (attr_type == ::oneflow::kAtComplexDouble) {
        // TODO(lml): use arrayattr to represent complex number is not safe, need improve.
        user_attr.mutable_at_complex_double();
        auto ref = attr.dyn_cast<ArrayAttr>();
        user_attr.mutable_at_complex_double()->set_real(
            ref.getValue()[0].dyn_cast<FloatAttr>().getValue().convertToDouble());
        user_attr.mutable_at_complex_double()->set_imag(
            ref.getValue()[1].dyn_cast<FloatAttr>().getValue().convertToDouble());
      } else {
        return failure();
      }
      (*user_conf->mutable_attr())[id.str()] = user_attr;
    }
  }
  return success();
}

LogicalResult ConvertUserOpAttributes(llvm::StringRef op_type_name, ValueRange operands,
                                      DictionaryAttr attributes, ::oneflow::OperatorConf& op_conf) {
  {
    std::vector<std::string> keys{};
    std::vector<int32_t> sizes{};
    if (failed(user_op::GetFilteredSegmentKeyAndSizes<OpTrait::AttrSizedOperandSegments>(
            op_type_name, operands.size(), attributes, keys, sizes))) {
      LOG(FATAL) << "fail to get filtered segment key and sizes";
      return failure();
    }
    for (const auto& s : keys) { op_conf.mutable_user_conf()->add_input_order(s); }
  }
  return doConvertUserOpAttributes(op_type_name, attributes, op_conf);
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
  return doConvertUserOpAttributes(op_type_name, op->getAttrDictionary(), op_conf);
}

LogicalResult ConvertUserOpAttributes(Operation* op, ::oneflow::OperatorConf& op_conf,
                                      bool is_mapping_size) {
  auto user_conf = op_conf.mutable_user_conf();
  std::string op_type_name = GetOpTypeName(op);
  op_conf.mutable_user_conf()->set_op_type_name(op_type_name);
  if (op->hasTrait<OpTrait::IsOpConfCompatible>()) {
    if (OpTrait::IsOpConfCompatible<void>::dump_attr(op, &op_conf).failed()) {
      return op->emitError("fail to save attr to op_conf");
    }
  }

  auto writeAttrToShape = [](mlir::Attribute& attr, ::oneflow::ShapeProto* shape) {
    for (auto v : attr.dyn_cast<ArrayAttr>().getValue()) {
      shape->add_dim(v.dyn_cast<IntegerAttr>().getSInt());
    }
  };

  auto writeAttrToStride = [](mlir::Attribute& attr, ::oneflow::Int64ListProto* stride) {
    for (auto v : attr.dyn_cast<ArrayAttr>().getValue()) {
      stride->add_dim(v.dyn_cast<IntegerAttr>().getSInt());
    }
  };

  for (auto id_attr : op->getAttrDictionary()) {
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
      const ::oneflow::AttrType attr_type = user_op::queryAttrType(op_type_name, attr_name);
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
        writeAttrToShape(attr, user_attr.mutable_at_shape());
      } else if (attr_type == ::oneflow::kAtStride) {
        writeAttrToStride(attr, user_attr.mutable_at_stride());
      } else if (attr_type == ::oneflow::kAtDataType) {
        const auto dt = support::FromMLIRAttrToOFDataType(attr);
        if (succeeded(dt)) {
          user_attr.set_at_data_type(dt.getValue());
        } else {
          op->emitError() << "fail to convert op attr to data type, key: " + id.str();
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
            op->emitError() << "fail to convert op attr to data type, key: " + id.str();
            return failure();
          }
        }
      } else if (attr_type == ::oneflow::kAtListShape) {
        for (auto shape_attr : attr.dyn_cast<ArrayAttr>().getValue()) {
          ::oneflow::ShapeProto* shape_ptr = user_attr.mutable_at_list_shape()->add_val();
          writeAttrToShape(shape_attr, shape_ptr);
        }
      } else if (attr_type == ::oneflow::kAtListStride) {
        for (auto stride_attr : attr.dyn_cast<ArrayAttr>().getValue()) {
          ::oneflow::Int64ListProto* stride_ptr = user_attr.mutable_at_list_stride()->add_val();
          writeAttrToStride(stride_attr, stride_ptr);
        }
      } else if (attr_type == ::oneflow::kAtListString) {
        // attr like nd_sbp requires the existence of list even it is empty
        user_attr.mutable_at_list_string();
        for (auto s : attr.dyn_cast<ArrayAttr>().getValue()) {
          user_attr.mutable_at_list_string()->add_val(s.dyn_cast<StringAttr>().getValue().str());
        }
      } else if (attr_type == ::oneflow::kAtComplexDouble) {
        // TODO(lml): use arrayattr to represent complex number is not safe, need improve.
        user_attr.mutable_at_complex_double();
        auto ref = attr.dyn_cast<ArrayAttr>();
        user_attr.mutable_at_complex_double()->set_real(
            ref.getValue()[0].dyn_cast<FloatAttr>().getValue().convertToDouble());
        user_attr.mutable_at_complex_double()->set_imag(
            ref.getValue()[1].dyn_cast<FloatAttr>().getValue().convertToDouble());
      } else {
        op->emitError() << "fail to convert op attr of name: " + attr_name;
        return failure();
      }
      (*user_conf->mutable_attr())[id.str()] = user_attr;
    }
  }
  {
    std::vector<std::string> keys{};
    std::vector<int32_t> sizes{};
    if (failed(user_op::GetFilteredSegmentKeyAndSizes<OpTrait::AttrSizedOperandSegments>(op, keys,
                                                                                         sizes))) {
      op->emitError("fail to convert user op input order");
      return failure();
    }
    for (const auto& s : keys) { op_conf.mutable_user_conf()->add_input_order(s); }

    if (is_mapping_size) {
      for (const auto it : llvm::zip(keys, sizes)) {
        auto key = std::get<0>(it).c_str();
        auto size = std::get<1>(it);
        auto tar = op_conf.mutable_user_conf()->mutable_input();
        auto val = ::oneflow::UserOpConf_ListString::default_instance();
        tar->insert({key, val});
        for (int i = 0; i < size; ++i) { tar->at(key).add_s(); }
      }
    }
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
    if (is_mapping_size) {
      for (const auto it : llvm::zip(keys, sizes)) {
        auto key = std::get<0>(it).c_str();
        auto size = std::get<1>(it);
        auto tar = op_conf.mutable_user_conf()->mutable_output();
        auto val = ::oneflow::UserOpConf_ListString::default_instance();
        tar->insert({key, val});
        for (int i = 0; i < size; ++i) { tar->at(key).add_s(); }
      }
    }
  }
  return success();
}
LogicalResult ConvertUserOpInputs(llvm::StringRef op_type_name, ValueRange operands,
                                  DictionaryAttr attributes, ::oneflow::UserOpConf* user_conf) {
  std::vector<std::string> keys{};
  std::vector<int32_t> sizes{};
  CHECK(user_op::GetFilteredSegmentKeyAndSizes<OpTrait::AttrSizedOperandSegments>(
            op_type_name, operands.size(), attributes, keys, sizes)
            .succeeded());
  int32_t input_idx = 0;
  for (auto tuple : llvm::zip(keys, sizes)) {
    auto input_key = std::get<0>(tuple);
    auto input_size = std::get<1>(tuple);
    for (int32_t i = 0; i < input_size; i++) {
      auto input_s_ptr = (*user_conf->mutable_input())[input_key].mutable_s()->Add();
      if (auto result = operands[input_idx].dyn_cast<mlir::OpResult>()) {
        *(input_s_ptr) = GetOutputLbn(result).getValue();
      } else if (auto argument = operands[input_idx].dyn_cast<mlir::BlockArgument>()) {
        *(input_s_ptr) = "BlockArgument/" + std::to_string(argument.getArgNumber());
      } else {
        LOG(FATAL) << "fail to convert MLIR result to protobuf, op_type_name: "
                          + op_type_name.str();
        return failure();
      }
      input_idx += 1;
    }
  }
  return success();
}

::oneflow::ShapeProto getAttrAsShape(mlir::Attribute& attr) {
  ::oneflow::ShapeProto shape{};
  for (auto v : attr.dyn_cast<ArrayAttr>().getValue()) {
    shape.add_dim(v.dyn_cast<IntegerAttr>().getSInt());
  }
  return shape;
}

::oneflow::Int64ListProto getAttrAsStride(mlir::Attribute& attr) {
  ::oneflow::Int64ListProto stride{};
  for (auto v : attr.dyn_cast<ArrayAttr>().getValue()) {
    stride.add_dim(v.dyn_cast<IntegerAttr>().getSInt());
  }
  return stride;
}

::oneflow::ParallelConf getParallelConfFromAttrDictionary(DictionaryAttr attributes) {
  ::oneflow::ParallelConf parallel_conf{};
  auto device_tag = attributes.get(OpTrait::IsOpConfCompatible<void>::getDeviceTagAttr())
                        .dyn_cast_or_null<StringAttr>();
  CHECK(device_tag) << "attr absent: "
                    << OpTrait::IsOpConfCompatible<void>::getDeviceTagAttr().str();
  parallel_conf.set_device_tag(device_tag.str());
  auto device_name = attributes.get(OpTrait::IsOpConfCompatible<void>::getDeviceNameAttr())
                         .dyn_cast_or_null<ArrayAttr>();
  CHECK(device_name) << "attr absent: "
                     << OpTrait::IsOpConfCompatible<void>::getDeviceNameAttr().str();
  for (auto s : device_name.getValue()) {
    parallel_conf.add_device_name(s.cast<StringAttr>().str());
  }
  if (auto hierarchy = attributes.get(OpTrait::IsOpConfCompatible<void>::getHierarchyAttr())
                           .dyn_cast_or_null<ArrayAttr>()) {
    for (auto dim : hierarchy.getValue()) {
      parallel_conf.mutable_hierarchy()->add_dim(dim.template dyn_cast<IntegerAttr>().getInt());
    }
  }
  return parallel_conf;
}

::oneflow::ParallelConf getParallelConfFromAttrs(Attribute device_name_attr,
                                                 Attribute device_tag_attr) {
  ::oneflow::ParallelConf parallel_conf{};
  auto device_tag = device_tag_attr.dyn_cast_or_null<StringAttr>();
  CHECK(device_tag) << "attr absent: "
                    << OpTrait::IsOpConfCompatible<void>::getDeviceTagAttr().str();
  parallel_conf.set_device_tag(device_tag.str());
  auto device_name = device_name_attr.dyn_cast_or_null<ArrayAttr>();
  CHECK(device_name) << "attr absent: "
                     << OpTrait::IsOpConfCompatible<void>::getDeviceNameAttr().str();
  for (auto s : device_name.getValue()) {
    parallel_conf.add_device_name(s.cast<StringAttr>().str());
  }
  return parallel_conf;
}

::oneflow::DeviceType getDeviceTypeFromAttrDictionary(DictionaryAttr attributes) {
  ::oneflow::ParallelConf parallel_conf{};
  auto device_tag = attributes.get(OpTrait::IsOpConfCompatible<void>::getDeviceTagAttr())
                        .dyn_cast_or_null<StringAttr>();
  CHECK(device_tag) << "attr absent: "
                    << OpTrait::IsOpConfCompatible<void>::getDeviceTagAttr().str();
  if (device_tag.str() == "cpu") {
    return ::oneflow::DeviceType::kCPU;
  } else if (device_tag.str() == "cuda") {
    return ::oneflow::DeviceType::kCUDA;
  } else {
    LOG(FATAL) << "unsupported device tag: " << device_tag.str();
    return ::oneflow::DeviceType::kInvalidDevice;
  }
}

::oneflow::AttrType queryAttrType(const std::string& op_type_name, const std::string& attr_name) {
  ::oneflow::user_op::UserOpDefWrapper op_def(support::getUserOpDef(op_type_name));
  CHECK(op_def.IsAttrName(attr_name)) << attr_name << " not a attr name for op: " << op_type_name;
  return op_def.GetAttrType(attr_name);
}

}  // namespace user_op

}  // namespace oneflow

}  // namespace mlir
