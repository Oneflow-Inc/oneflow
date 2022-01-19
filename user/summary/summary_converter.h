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
#ifndef ONEFLOW_USER_SUMMARY_SUMMARY_CONVERTER_H_
#define ONEFLOW_USER_SUMMARY_SUMMARY_CONVERTER_H_

#include "nlohmann/json.hpp"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

namespace summary {

static void ConvertProtobufMsg2Json(nlohmann::json& json_value, const PbMessage& pb_msg);

static void ConvertRepeatedField2Json(nlohmann::json& json_value, const PbMessage& pb_msg,
                                      const PbFd* pb_field,
                                      const google::protobuf::Reflection* pb_reflection) {
  if (NULL == pb_field || NULL == pb_reflection) { ConvertProtobufMsg2Json(json_value, pb_msg); }

  for (int i = 0; i < pb_reflection->FieldSize(pb_msg, pb_field); ++i) {
    nlohmann::json tmp_json_value;
    switch (pb_field->type()) {
      case PbFd::TYPE_MESSAGE: {
        const PbMessage& msg = pb_reflection->GetRepeatedMessage(pb_msg, pb_field, i);
        if (0 != msg.ByteSize()) { ConvertProtobufMsg2Json(tmp_json_value, msg); }
      } break;
      case PbFd::TYPE_INT32:
        tmp_json_value = pb_reflection->GetRepeatedInt32(pb_msg, pb_field, i);
        break;
      case PbFd::TYPE_UINT32:
        tmp_json_value = pb_reflection->GetRepeatedUInt32(pb_msg, pb_field, i);
        break;
      case PbFd::TYPE_INT64: {
        static char int64_str[25];
        memset(int64_str, 0, sizeof(int64_str));
        snprintf(int64_str, sizeof(int64_str), "%lld",
                 (long long)pb_reflection->GetRepeatedInt64(pb_msg, pb_field, i));
        tmp_json_value = int64_str;
      } break;
      case PbFd::TYPE_UINT64: {
        static char uint64str[25];
        memset(uint64str, 0, sizeof(uint64str));
        snprintf(uint64str, sizeof(uint64str), "%llu",
                 (unsigned long long)pb_reflection->GetRepeatedUInt64(pb_msg, pb_field, i));
        tmp_json_value = uint64str;
      } break;
      case PbFd::TYPE_STRING:
      case PbFd::TYPE_BYTES:
        tmp_json_value = pb_reflection->GetRepeatedString(pb_msg, pb_field, i);
        break;
      case PbFd::TYPE_BOOL:
        tmp_json_value = pb_reflection->GetRepeatedBool(pb_msg, pb_field, i);
        break;
      case PbFd::TYPE_ENUM:
        tmp_json_value = pb_reflection->GetRepeatedEnum(pb_msg, pb_field, i)->name();
        break;
      case PbFd::TYPE_FLOAT:
        tmp_json_value = pb_reflection->GetRepeatedFloat(pb_msg, pb_field, i);
        break;
      case PbFd::TYPE_DOUBLE:
        tmp_json_value = pb_reflection->GetRepeatedDouble(pb_msg, pb_field, i);
        break;
      default: break;
    }
    json_value.emplace_back(tmp_json_value);
  }
}

static void ConvertProtobufMsg2Json(nlohmann::json& json_value, const PbMessage& pb_msg) {
  const google::protobuf::Descriptor* pb_descriptor = pb_msg.GetDescriptor();
  const google::protobuf::Reflection* pb_reflection = pb_msg.GetReflection();

  const int count = pb_descriptor->field_count();

  for (int i = 0; i < count; ++i) {
    const PbFd* pb_field = pb_descriptor->field(i);

    if (pb_field->is_repeated()) {
      if (pb_reflection->FieldSize(pb_msg, pb_field) > 0) {
        ConvertRepeatedField2Json(json_value[pb_field->name()], pb_msg, pb_field, pb_reflection);
      }
      continue;
    }

    if (!pb_reflection->HasField(pb_msg, pb_field)) { continue; }

    switch (pb_field->type()) {
      case PbFd::TYPE_MESSAGE: {
        const PbMessage& msg = pb_reflection->GetMessage(pb_msg, pb_field);
        if (0 != msg.ByteSize()) { ConvertProtobufMsg2Json(json_value[pb_field->name()], msg); }
      } break;
      case PbFd::TYPE_INT32:
        json_value[pb_field->name()] = pb_reflection->GetInt32(pb_msg, pb_field);
        break;
      case PbFd::TYPE_UINT32:
        json_value[pb_field->name()] = pb_reflection->GetUInt32(pb_msg, pb_field);
        break;
      case PbFd::TYPE_INT64: {
        static char int64_str[25];
        memset(int64_str, 0, sizeof(int64_str));
        snprintf(int64_str, sizeof(int64_str), "%lld",
                 (long long)pb_reflection->GetInt64(pb_msg, pb_field));
        json_value[pb_field->name()] = int64_str;
      } break;
      case PbFd::TYPE_UINT64: {
        static char uint64_str[25];
        memset(uint64_str, 0, sizeof(uint64_str));
        snprintf(uint64_str, sizeof(uint64_str), "%llu",
                 (unsigned long long)pb_reflection->GetUInt64(pb_msg, pb_field));
        json_value[pb_field->name()] = uint64_str;
      } break;
      case PbFd::TYPE_STRING:
      case PbFd::TYPE_BYTES: {
        json_value[pb_field->name()] = pb_reflection->GetString(pb_msg, pb_field);
      } break;
      case PbFd::TYPE_BOOL: {
        json_value[pb_field->name()] = pb_reflection->GetBool(pb_msg, pb_field);
      } break;
      case PbFd::TYPE_ENUM: {
        json_value[pb_field->name()] = pb_reflection->GetEnum(pb_msg, pb_field)->name();
      } break;
      case PbFd::TYPE_FLOAT: {
        json_value[pb_field->name()] = pb_reflection->GetFloat(pb_msg, pb_field);
      } break;
      case PbFd::TYPE_DOUBLE: {
        json_value[pb_field->name()] = pb_reflection->GetDouble(pb_msg, pb_field);
      } break;
      default: break;
    }
  }
}

}  // namespace summary
}  // namespace oneflow

#endif  // ONEFLOW_USER_SUMMARY_SUMMARY_CONVERTER_H_
