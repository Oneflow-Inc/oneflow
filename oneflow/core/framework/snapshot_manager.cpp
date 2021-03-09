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
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/persistence/file_system.h"

namespace oneflow {

void SnapshotManager::Load(const std::string& root_dir, bool refresh) {
  std::shared_ptr<fs::FileSystem> fs(LocalFS());
  CHECK(fs->IsDirectory(root_dir));
  if (refresh) { variable_name2path_.clear(); }
  for (const auto& dir : fs->ListDir(root_dir)) {
    std::string absolute_dir = JoinPath(root_dir, dir);
    if (!fs->IsDirectory(absolute_dir)) { continue; }
    for (const auto& file : fs->ListDir(absolute_dir)) {
      std::string absolute_file = JoinPath(absolute_dir, file);
      if (file == "out" && fs->IsRegularFile(absolute_file)) {
        variable_name2path_[file] = absolute_file;
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
