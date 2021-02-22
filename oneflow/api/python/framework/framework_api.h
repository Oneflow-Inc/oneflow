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
#ifndef ONEFLOW_API_PYTHON_FRAMEWORK_FRAMEWORK_API_H_
#define ONEFLOW_API_PYTHON_FRAMEWORK_FRAMEWORK_API_H_

#include "oneflow/api/python/framework/framework.h"
#include "oneflow/core/serving/saved_model.cfg.h"

inline void RegisterForeignCallbackOnlyOnce(oneflow::ForeignCallback* callback) {
  return oneflow::RegisterForeignCallbackOnlyOnce(callback).GetOrThrow();
}

inline void RegisterWatcherOnlyOnce(oneflow::ForeignWatcher* watcher) {
  return oneflow::RegisterWatcherOnlyOnce(watcher).GetOrThrow();
}

inline void RegisterBoxingUtilOnlyOnce(oneflow::ForeignBoxingUtil* boxing_util) {
  return oneflow::RegisterBoxingUtilOnlyOnce(boxing_util).GetOrThrow();
}

inline void LaunchJob(const std::shared_ptr<oneflow::ForeignJobInstance>& cb) {
  return oneflow::LaunchJob(cb).GetOrThrow();
}

inline std::string GetSerializedInterUserJobInfo() {
  return oneflow::GetSerializedInterUserJobInfo().GetOrThrow();
}

inline std::string GetSerializedJobSet() { return oneflow::GetSerializedJobSet().GetOrThrow(); }

inline std::string GetSerializedStructureGraph() {
  return oneflow::GetSerializedStructureGraph().GetOrThrow();
}

inline std::string GetSerializedCurrentJob() {
  return oneflow::GetSerializedCurrentJob().GetOrThrow();
}

inline std::string GetFunctionConfigDef() { return oneflow::GetFunctionConfigDef().GetOrThrow(); }

inline std::string GetScopeConfigDef() { return oneflow::GetScopeConfigDef().GetOrThrow(); }

inline std::string GetMachine2DeviceIdListOFRecordFromParallelConf(
    const std::string& parallel_conf) {
  return oneflow::GetSerializedMachineId2DeviceIdListOFRecord(parallel_conf).GetOrThrow();
}

inline std::shared_ptr<::oneflow::cfg::SavedModel> LoadSavedModel(
    const std::string& saved_model_meta_file, bool is_prototxt_file) {
  return oneflow::LoadSavedModel(saved_model_meta_file, is_prototxt_file).GetPtrOrThrow();
}

inline void LoadLibraryNow(const std::string& lib_path) {
  return oneflow::LoadLibraryNow(lib_path).GetOrThrow();
}

#endif  // ONEFLOW_API_PYTHON_FRAMEWORK_FRAMEWORK_API_H_
