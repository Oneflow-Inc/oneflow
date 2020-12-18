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

std::shared_ptr<oneflow::cfg::ErrorProto> RegisterForeignCallbackOnlyOnce(
    oneflow::ForeignCallback* callback) {
  return oneflow::RegisterForeignCallbackOnlyOnce(callback).GetDataAndErrorProto();
}

std::shared_ptr<oneflow::cfg::ErrorProto> RegisterWatcherOnlyOnce(
    oneflow::ForeignWatcher* watcher) {
  return oneflow::RegisterWatcherOnlyOnce(watcher).GetDataAndErrorProto();
}

std::shared_ptr<oneflow::cfg::ErrorProto> LaunchJob(
    const std::shared_ptr<oneflow::ForeignJobInstance>& cb) {
  return oneflow::LaunchJob(cb).GetDataAndErrorProto();
}
#endif  // ONEFLOW_ONEFLOW_API_PYTHON_FRAMEWORK_FRAMEWORK_H_
