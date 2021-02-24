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
#include <sys/stat.h>
#include <memory>

#include "oneflow/core/common/platform.h"
#include "oneflow/core/common/file_system.h"

#if defined(OF_PLATFORM_WINDOWS)
#include <windows.h>
static constexpr char _delimiter = '\\';
typedef struct _stat stat;
void stat(const char* fname, struct stat* buf) { _stat(fname, buf); };
#elif defined(OF_PLATFORM_POSIX)
#include <dirent.h>
#include <sys/types.h>
static constexpr char _delimiter = '/';
#else
static_assert(0, "File system only support windows and posix platform.");
#endif

#ifndef S_ISDIR
#define S_ISDIR(mode) (((mode) & _S_IFMT) == _S_IFDIR)
#endif  // S_ISDIR
#ifndef S_ISREG
#define S_ISREG(mode) (((mode) & _S_IFMT) == _S_IFDIR)
#endif  // S_ISREG

namespace oneflow {
namespace file_system {

std::string Path::basename() const {
  auto pos = path_.find_first_of(_delimiter);
  if (pos == std::string::npos) { return path_; }
  return path_.substr(pos + 1);
}

std::string Path::dirname() const {
  auto pos = path_.find_first_of(_delimiter);
  if (pos == std::string::npos) { return ""; }
  return path_.substr(0, pos);
}

class Stat {
 public:
  Stat() = default;
  explicit Stat(const std::string& fname) {
    stat_.reset(new struct stat);
    if (stat(fname.c_str(), stat_.get()) != 0) { stat_.reset(); }
  }

  bool exists() const { return stat_.get(); }

  bool is_directory() const {
    if (stat_.get()) { return S_ISDIR(stat_->st_mode); }
    return false;
  }

  bool is_regular_file() const {
    if (stat_.get()) { return S_ISREG(stat_->st_mode); }
    return false;
  }

 private:
  std::shared_ptr<struct stat> stat_;
};

bool Path::exists() const {
  Stat stat(path_);
  return stat.exists();
}

bool Path::is_directory() const {
  Stat stat(path_);
  return stat.is_directory();
}

bool Path::is_regular_file() const {
  Stat stat(path_);
  return stat.is_regular_file();
}

Path& Path::join(const Path& rhs) {
  path_ = path_ + _delimiter + rhs.path_;
  return *this;
}

std::string basename(const std::string& path) { return Path(path).basename(); }
std::string dirname(const std::string& path) { return Path(path).dirname(); }

bool is_directory(const std::string& path) { return Path(path).is_directory(); }
bool is_regular_file(const std::string& path) { return Path(path).is_regular_file(); }

std::string join_path(const std::string& lhs, const std::string& rhs) {
  return Path(lhs).join(Path(rhs)).path();
}

std::vector<std::string> walk(const std::string& path) { return walk(Path(path)); }

#if defined(OF_PLATFORM_WINDOWS)
std::vector<std::string> walk(const Path& path) {
  std::vector<std::string> walked_paths;
  // Refer to cppfs: source/windows/LocalFileHandle.cpp
  WIN32_FIND_DATA findData;
  std::string query = join_path(path.path(), "*");
  HANDLE findHandle = FindFirstFileA(query.c_str(), &findData);
  if (findHandle == INVALID_HANDLE_VALUE) { return std::move(walked_paths); }
  do {
    const std::string& fname = findData.cFileName;
    if (fname != ".." && fname != ".") { walked_paths.push_back(fname); }
  } while (FindNextFile(findHandle, &findData));

  FindClose(findHandle);
  return std::move(walked_paths);
}
#else
std::vector<std::string> walk(const Path& path) {
  std::vector<std::string> walked_paths;
  // Refer to cppfs: source/posix/LocalFileHandle.cpp
  DIR* dir = opendir(path.path().c_str());
  if (!dir) { return std::move(walked_paths); }
  struct dirent* entry = readdir(dir);
  while (entry) {
    const std::string& fname = entry->d_name;
    if (fname != ".." && fname != ".") { walked_paths.push_back(fname); }
    entry = readdir(dir);
  }
  closedir(dir);
  return std::move(walked_paths);
}
#endif
}  // namespace file_system
}  // namespace oneflow
