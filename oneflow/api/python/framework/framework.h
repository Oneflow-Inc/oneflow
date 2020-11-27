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
#ifndef ONEFLOW_ONEFLOW_API_PYTHON_FRAMEWORK_FRAMEWORK_H_
#define ONEFLOW_ONEFLOW_API_PYTHON_FRAMEWORK_FRAMEWORK_H_

#include <string>
#include "oneflow/api/python/framework/framework_helper.h"

std::string RegisterForeignCallbackOnlyOnce(oneflow::ForeignCallback* callback) {
  std::string error_str;
  oneflow::RegisterForeignCallbackOnlyOnce(callback).GetDataAndSerializedErrorProto(&error_str);
  return error_str;
}

std::string RegisterWatcherOnlyOnce(oneflow::ForeignWatcher* watcher) {
  std::string error_str;
  oneflow::RegisterWatcherOnlyOnce(watcher).GetDataAndSerializedErrorProto(&error_str);
  return error_str;
}

std::string LaunchJob(const std::shared_ptr<oneflow::ForeignJobInstance>& cb) {
  std::string error_str;
  oneflow::LaunchJob(cb).GetDataAndSerializedErrorProto(&error_str);
  return error_str;
}
#endif  // ONEFLOW_ONEFLOW_API_PYTHON_FRAMEWORK_FRAMEWORK_H_
