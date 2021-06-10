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
#include "oneflow/core/persistence/file_system.h"
#include <errno.h>
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/core/persistence/hadoop/hadoop_file_system.h"
#include "oneflow/core/persistence/posix/posix_file_system.h"
#include "oneflow/core/job/job_set.pb.h"

namespace oneflow {

namespace fs {

void FileSystem::CreateDirIfNotExist(const std::string& dirname) {
  if (IsDirectory(dirname)) { return; }
  CreateDir(dirname);
}

void FileSystem::RecursivelyCreateDirIfNotExist(const std::string& dirname) {
  if (IsDirectory(dirname)) { return; }
  RecursivelyCreateDir(dirname);
}

bool FileSystem::IsDirEmpty(const std::string& dirname) { return ListDir(dirname).empty(); }

std::string FileSystem::TranslateName(const std::string& name) const { return CleanPath(name); }

void FileSystem::MakeEmptyDir(const std::string& dirname) {
  if (IsDirectory(dirname)) { RecursivelyDeleteDir(dirname); }
  RecursivelyCreateDir(dirname);
}

void FileSystem::RecursivelyDeleteDir(const std::string& dirname) {
  CHECK(FileExists(dirname));
  std::deque<std::string> dir_q;      // Queue for the BFS
  std::vector<std::string> dir_list;  // List of all dirs discovered
  dir_q.push_back(dirname);
  // ret : Status to be returned.
  // Do a BFS on the directory to discover all the sub-directories. Remove all
  // children that are files along the way. Then cleanup and remove the
  // directories in reverse order.;
  while (!dir_q.empty()) {
    std::string dir = dir_q.front();
    dir_q.pop_front();
    dir_list.push_back(dir);
    // GetChildren might fail if we don't have appropriate permissions.
    std::vector<std::string> children = ListDir(dir);
    for (const std::string& child : children) {
      const std::string child_path = JoinPath(dir, child);
      // If the child is a directory add it to the queue, otherwise delete it.
      if (IsDirectory(child_path)) {
        dir_q.push_back(child_path);
      } else {
        // Delete file might fail because of permissions issues or might be
        // unimplemented.
        DelFile(child_path);
      }
    }
  }
  // Now reverse the list of directories and delete them. The BFS ensures that
  // we can delete the directories in this order.
  std::reverse(dir_list.begin(), dir_list.end());
  for (const std::string& dir : dir_list) {
    // Delete dir might fail because of permissions issues or might be
    // unimplemented.
    DeleteDir(dir);
  }
}

void FileSystem::RecursivelyCreateDir(const std::string& dirname) {
  std::string remaining_dir = dirname;
  std::vector<std::string> sub_dirs;
  while (!remaining_dir.empty()) {
    bool status = FileExists(remaining_dir);
    if (status) { break; }
    // Basename returns "" for / ending dirs.
    if (remaining_dir[remaining_dir.length() - 1] != '/') {
      sub_dirs.push_back(Basename(remaining_dir));
    }
    remaining_dir = Dirname(remaining_dir);
  }

  // sub_dirs contains all the dirs to be created but in reverse order.
  std::reverse(sub_dirs.begin(), sub_dirs.end());

  // Now create the directories.
  std::string built_path = remaining_dir;
  for (const std::string& sub_dir : sub_dirs) {
    built_path = JoinPath(built_path, sub_dir);
    CreateDir(built_path);
  }
}

}  // namespace fs

fs::FileSystem* LocalFS() {
#ifdef OF_PLATFORM_POSIX
  static fs::FileSystem* fs = new fs::PosixFileSystem;
#endif
  return fs;
}

fs::FileSystem* NetworkFS() { return LocalFS(); }

fs::FileSystem* HadoopFS(const HdfsConf& hdfs_conf) {
  static fs::FileSystem* fs = new fs::HadoopFileSystem(hdfs_conf);
  return fs;
}

fs::FileSystem* GetFS(const FileSystemConf& file_system_conf) {
  if (file_system_conf.has_localfs_conf()) {
    return LocalFS();
  } else if (file_system_conf.has_networkfs_conf()) {
    return NetworkFS();
  } else if (file_system_conf.has_hdfs_conf()) {
    return HadoopFS(file_system_conf.hdfs_conf());
  } else {
    UNIMPLEMENTED();
  }
}

// Will be deprecated
fs::FileSystem* DataFS() { return GetFS(Global<const IOConf>::Get()->data_fs_conf()); }
fs::FileSystem* DataFS(int64_t session_id) {
  return GetFS(Global<const IOConf>::Get(session_id)->data_fs_conf());
}
fs::FileSystem* SnapshotFS() { return GetFS(Global<const IOConf>::Get()->snapshot_fs_conf()); }

void CreateLocalFS(std::unique_ptr<fs::FileSystem>& fs) {
#ifdef OF_PLATFORM_POSIX
  fs.reset(new fs::PosixFileSystem);
#else
  OF_UNIMPLEMENTED();
#endif
}

void CreateHadoopFS(std::unique_ptr<fs::FileSystem>& fs, const std::string& namenode) {
  fs.reset(new fs::HadoopFileSystem(namenode));
}

void CreateFileSystemFromEnv(std::unique_ptr<fs::FileSystem>& fs, const std::string& env_prefix) {
  CHECK(!fs);

  auto fs_type_env = env_prefix + "_TYPE";
  const char* fs_type = std::getenv(fs_type_env.c_str());
  std::string fs_type_str;
  if (fs_type) {
    fs_type_str = ToLower(fs_type);
  } else {
    // local file system by default
    fs_type_str = "local";
  }

  if (fs_type_str == "local") {
    CreateLocalFS(fs);
  } else if (fs_type_str == "network") {
    CreateLocalFS(fs);
  } else if (fs_type_str == "hdfs") {
    auto hdfs_nn_env = env_prefix + "_HDFS_NAMENODE";
    const char* hdfs_namenode = std::getenv(hdfs_nn_env.c_str());
    if (hdfs_namenode == nullptr) {
      LOG(FATAL) << "env " << hdfs_nn_env << " must be set when " << fs_type_env
                 << " be set to hdfs";
    }
    CreateHadoopFS(fs, hdfs_namenode);
  } else {
    CreateLocalFS(fs);
    LOG(WARNING) << "invalid env " << fs_type_env << " " << fs_type
                 << ", return the local file system instead";
  }
}

// New data & snapshot file system inferface
fs::FileSystem* GetDataFS() {
  static std::unique_ptr<fs::FileSystem> data_fs;
  static std::mutex data_fs_mutex;
  {
    std::lock_guard<std::mutex> lock(data_fs_mutex);
    if (!data_fs) { CreateFileSystemFromEnv(data_fs, "ONEFLOW_DATA_FILE_SYSTEM"); }
    CHECK(data_fs);
  }
  return data_fs.get();
}

fs::FileSystem* GetSnapshotFS() {
  static std::unique_ptr<fs::FileSystem> snapshot_fs;
  static std::mutex snapshot_fs_mutex;
  {
    std::lock_guard<std::mutex> lock(snapshot_fs_mutex);
    if (!snapshot_fs) { CreateFileSystemFromEnv(snapshot_fs, "ONEFLOW_SNAPSHOT_FILE_SYSTEM"); }
    CHECK(snapshot_fs);
  }
  return snapshot_fs.get();
}

}  // namespace oneflow
