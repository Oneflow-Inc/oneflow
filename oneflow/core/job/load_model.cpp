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
#include "oneflow/core/job/load_model.h"
#include <string>
#include "oneflow/core/job/saved_model.pb.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/str_util.h"

namespace oneflow {

namespace {

void LoadSavedModelProtoFromFile(const LoadModelProto& load_model_proto,
                                 SavedModel* saved_model_proto) {
  std::string saved_model_proto_path =
      JoinPath(load_model_proto.model_path(), std::to_string(load_model_proto.version()),
               "saved_model.prototxt");
  ParseProtoFromTextFile(saved_model_proto_path, saved_model_proto);
}

Maybe<void> LoadModel(const std::string& load_model_proto_str) {
  LoadModelProto load_model_proto;
  CHECK_OR_RETURN(TxtString2PbMessage(load_model_proto_str, &load_model_proto))
      << "load_model_proto parse failed";
  SavedModel saved_model_proto;
  LoadSavedModelProtoFromFile(load_model_proto, &saved_model_proto);
  std::cout << "saved model:\n" << PbMessage2TxtString(saved_model_proto) << std::endl;

  return Maybe<void>::Ok();
}

}  // namespace

std::string ApiLoadModel(const std::string& load_model_proto_str) {
  std::string error_str;
  LoadModel(load_model_proto_str).GetDataAndSerializedErrorProto(&error_str);
  return error_str;
}

}  // namespace oneflow
