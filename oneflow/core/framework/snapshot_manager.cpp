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
#include "oneflow/core/framework/snapshot_manager.h"
#include "oneflow/core/common/file_system.h"

namespace oneflow {

void SnapshotManager::Load(const std::string& root_dir, bool refresh) {
  CHECK(file_system::is_directory(root_dir));
  if (refresh) { variable_name2path_.clear(); }
  for (const auto& dir : file_system::walk(root_dir)) {
    if (!file_system::is_directory(dir)) { continue; }
    std::string dir_basename = file_system::basename(dir);
    for (const auto& file : file_system::walk(dir)) {
      std::string file_basename = file_system::basename(file);
      if (file_basename == "out" && file_system::is_regular_file(file)) {
        variable_name2path_[dir_basename] = file;
      }
    }
  }
}

const std::string& SnapshotManager::GetSnapshotPath(const std::string& variable_name) const {
  const auto& it = variable_name2path_.find(variable_name);
  if (it != variable_name2path_.end()) {
    return it->second;
  } else {
    return default_path_;
  }
}

}  // namespace oneflow
